import pandas as pd
from pathlib import Path
import ssl

# Bypassing a MacOS specific security check
ssl._create_default_https_context = ssl._create_unverified_context

# Pandas brings in the data from the API and saves it into a dataframe
url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
df = pd.read_csv(url)

print(f"No. of stars downloaded: {len(df)}")


# Out of several data columns (such as telescopic parameters, stellar coordinates etc), only include the required columns
df = df[['TIC ID', 'TFOPWG Disposition', 'Period (days)', 'Epoch (BJD)']]
df = df.dropna(subset=['TFOPWG Disposition'])

""" TFOPWG Dispositions are as follows-
KP = Known Planet
PC = Planetary Candidate
CP = Candidate Planet

FP = False Positive (eclipsing binary system, asteriod flyby)
FA = False Alarm (systemic errors)
"""
def map_nasa_labels(disposition):
	if disposition in ['KP', 'CP', 'PC']:
		return 1
	elif disposition in ['FP', 'FA']:
		return 0
	else:
		return None

# Apply the mapping function and drop any stars that haven't been classified yet
df['final_label'] = df['TFOPWG Disposition'].apply(map_nasa_labels)
df = df.dropna(subset = ['final_label'])
df['final_label'] = df['final_label'].astype(int)

# Perform schema matching
df = df.rename(columns = {'TIC ID': 'TIC_ID'})


# Set seed to 42 for reproducibility
num_planets = len(df[df['final_label'] == 1])
num_junk = len(df[df['final_label'] == 0])
sample_size = min(2500, num_planets, num_junk)

print(f"Found {num_planets} planets and {num_junk} others")

planets = df[df['final_label'] == 1].sample(sample_size, random_state = 42)
junk = df[df['final_label'] == 0].sample(sample_size, random_state = 42)

df_balanced = pd.concat([planets, junk]).sample(frac = 1, random_state = 42).reset_index(drop = True)


# Save to data folder
out_path = Path("data/processed_labels.csv")
out_path.parent.mkdir(exist_ok = True)

df_balanced[['TIC_ID', 'final_label', 'Period (days)', 'Epoch (BJD)']].to_csv(out_path, index = False)

print("Head: ")
print(df_balanced[['TIC_ID', 'TFOPWG Disposition', 'Period (days)', 'Epoch (BJD)', 'final_label']].head())
