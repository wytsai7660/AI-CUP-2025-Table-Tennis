import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import math

@dataclass
class Configs:
    def __init__(self):
        self.seq_len = 1024
        self.patch_size = 8
        self.in_chans = 6
        self.n_embd = 128
        self.n_layer = 4
        self.dropout = 0.2
        self.out_size = 21
        self.bias = False
        self.pos_embed = False  # Set to True to enable positional embedding
        self.use_wandb = False  # Set to True to enable Weights & Biases logging
        

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

        
class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        assert seq_len % patch_size == 0, "Sequence length must be divisible by patch size."
        self.num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_chans, embed_dim, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        x = x.permute(0, 2, 1)
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size) # (B, C, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, x.size(1), -1)
        x = self.proj(x) # (B, num_patches, embed_dim)
        
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SwingGRU(nn.Module):
    def __init__(self, configs: Configs):
        super(SwingGRU, self).__init__()
        self.configs = configs
        if configs.pos_embed:
            self.pos_embed = PositionalEmbedding(configs.n_embd, max_len=(configs.seq_len // configs.patch_size) + 1)  # +1 for the class token
        # self.what_embed = nn.Linear(configs.n_embd, configs.n_embd, bias=configs.bias)  # Adjusting the positional embedding to match the embedding dimension
        self.patch_embed = PatchEmbed1D(configs.seq_len, configs.patch_size, configs.in_chans, configs.n_embd)
        # self.pre_norm = LayerNorm(configs.n_embd, bias=configs.bias)
        self.embedding_dropout = nn.Dropout(configs.dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, configs.n_embd))  # Class token for classification tasks
                
        self.shared_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=configs.n_embd,
                nhead=8,
                dim_feedforward=4 * configs.n_embd,  # Feedforward dimension
                dropout=configs.dropout,
                activation=F.gelu,  # Activation function
                norm_first=True,  # Normalize before the attention and feedforward layers
                batch_first=True,
                bias=configs.bias,
            ),
            num_layers=configs.n_layer,
            enable_nested_tensor=False,
            norm=LayerNorm(configs.n_embd, bias=configs.bias),
        )
        
              
        self.predict_head  = nn.Sequential(
            nn.Linear(configs.n_embd, configs.n_embd, bias=configs.bias),  # Project to embedding size
            LayerNorm(configs.n_embd, bias=configs.bias),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.n_embd, configs.out_size, bias=configs.bias)  # Output layer for classification,
        )
            
    def forward(self, x, mode = None, target = None, criterions=None):
        
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        
        if self.configs.pos_embed:
            x = x + self.pos_embed(x)  # Add positional embeddings
        
        x = self.embedding_dropout(x)  # Apply dropout
        shared_transfomer_out = self.shared_transformer(x)  # (B, num_patches, embed_dim)
        logits = self.predict_head(shared_transfomer_out[:, 0, :])  # (B, out_size)
                        
        if target is None:
            for i, (start, end, criterion, weight) in enumerate(criterions):
                logits[:, start:end] = F.softmax(logits[:, start:end], dim=-1)
            return logits
        else:
            total_loss = 0.0
            losses = np.zeros(len(criterions), dtype=np.float32)
            for i, (start, end, criterion, weight) in enumerate(criterions):
                loss = criterion(logits[:, start:end], target[:, i].long())
                # total_loss += weight * loss
                total_loss += loss
                losses[i] = loss.item()
                logits[:, start:end] = F.softmax(logits[:, start:end], dim=-1)
            return logits, total_loss, losses
        
    def configure_optimizers(self, weight_decay, learning_rate, betas=(0.9, 0.999)):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        optimizer = torch.optim.RAdam(optim_groups, lr=learning_rate, betas=betas)

        return optimizer
