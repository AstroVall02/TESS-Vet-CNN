import pandas as pd
from pathlib import Path

csv_path = Path("data/processed_labels.csv")
df = pd.read_csv(csv_path)
org_count = len(df)

lightcurves_dir = Path("data/lightcurves")

def check_file(tic_id):
	file_path = lightcurves_dir / f"TIC_{tic_id}.npy"
	return file_path.exists()

df_cleaned = df[df['TIC_ID'].apply(check_file)]

df_cleaned.to_csv(csv_path, index = False)

print(f"Original count = {org_count}")
print(f"New count = {len(df_cleaned)}")
