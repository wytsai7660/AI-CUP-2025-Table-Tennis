import pandas as pd
import numpy as np
from pathlib import Path
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
import pickle
from tqdm import tqdm

with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

scaler = model['scaler']
gender_clf = model['gender']
hand_clf = model['hand']
years_clf = model['years']
level_clf = model['level']



info_test = pd.read_csv('data/test/test_info.csv')
test_unique_ids = set(info_test['unique_id']) # Use a set for faster lookup

datapath_test = 'data/tabular_data_test'
datalist_test = list(Path(datapath_test).glob('**/*.csv'))

results = []

print(f"predicting {len(datalist_test)} files...")

le_gender_classes = gender_clf.classes_
le_hand_classes = hand_clf.classes_
le_years_classes = years_clf.classes_
le_level_classes = level_clf.classes_


for i, file in tqdm(enumerate(datalist_test), leave=False, total=len(datalist_test)):
    unique_id = int(Path(file).stem)

    # if unique_id not in test_unique_ids:
    #     continue

    data = pd.read_csv(file)

    if data.empty:
        num_years_classes = len(le_years_classes)
        num_level_classes = len(le_level_classes)
        result = {
            'unique_id': unique_id,
            'gender': 0.5,
            'hold racket handed': 0.5,
            'play years_0': 0.5,
            'play years_1': 0.5,
            'play years_2': 0.5,
            'level_2': 0.5,
            'level_3': 0.5,
            'level_4': 0.5,
            'level_5': 0.5,
        }
        expected_cols = ['unique_id', 'gender', 'hold racket handed', 'play years_0', 'play years_1', 'play years_2', 'level_2', 'level_3', 'level_4', 'level_5']
        for col in expected_cols:
             if col not in result:
                 result[col] = np.nan
        results.append(result)
        continue
    
    data_scaled = scaler.transform(data)


    pred_hand_proba = hand_clf.predict_proba(data_scaled)
    avg_hand_probs = np.mean(pred_hand_proba, axis=0)
    chosen_class = np.argmax(avg_hand_probs)  # 0 或 1
    avg_hand_prob_class0 = np.mean(pred_hand_proba[:, chosen_class])

    result = {
        'unique_id': unique_id,
        'gender': 0.5,
        'hold racket handed': avg_hand_prob_class0,
        'play years_0': 0.5,
        'play years_1': 0.5,
        'play years_2': 0.5,
        'level_2': 0.5,
        'level_3': 0.5,
        'level_4': 0.5,
        'level_5': 0.5,
    }
    results.append(result)

results_df = pd.DataFrame(results)

output_columns = [
    'unique_id',
    'gender',
    'hold racket handed',
    'play years_0', 'play years_1', 'play years_2',
    'level_2', 'level_3', 'level_4', 'level_5'
]

for col in output_columns:
    if col not in results_df.columns:
        results_df[col] = 0.5

results_df = results_df[output_columns]

results_df['unique_id'] = results_df['unique_id'].astype(int)

output_filename = 'submission.csv'
results_df.to_csv(output_filename, index=False, float_format='%.8f') # 使用較高精度儲存浮點數

print(f"Results saved to {output_filename}")