import pandas as pd

csvfiles = ["Gender/gender.csv", "Hold Racket Hand/hand.csv", 
            "Play_years/play_years.csv", "Level/level.csv"]

columns_to_keep = {
    csvfiles[0]: ["unique_id", "gender"],
    csvfiles[1]: ["unique_id", "hold racket handed"],
    csvfiles[2]: ["unique_id", "play years_0", "play years_1", "play years_2"],
    csvfiles[3]: ["unique_id", "level_2", "level_3", "level_4", "level_5"]
}

dfs = []
for file in csvfiles:
    df = pd.read_csv(file)
    df_filtered = df[columns_to_keep[file]]
    dfs.append(df_filtered)

merged_df = dfs[0]
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='unique_id', how='outer')

merged_df.to_csv('merged_predictions.csv', index=False)
print("Merged CSV saved as 'merged_predictions.csv'")

