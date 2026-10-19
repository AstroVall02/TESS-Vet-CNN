import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import os

# Transforms used for Data Augmentation
class TESSTransform:
    #adds random noise and shifts the lightcurve data to address model overfitting issue
    
    def __init__(self, noise_level = 0.001, shift_max = 200, seed = None):
        self.noise_level = noise_level
        self.shift_max = shift_max
        
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        self.np_rng = np.random.RandomState(seed)

    def __call__(self, x):
        # x lies in the range [1, 20000]
        # Adding gausian noise-
        ## noise = torch.randn_like(x) * self.noise_level
        ## x += noise

        noise = torch.randn(x.shape, generator = self.generator) * self.noise_level
        x = x + noise

        # Random shifts to the data in the range [-max, max]
        ## shift = np.random.randint(-self.shift_max, self.shift_max)
        shift = self.np_rng.randint(-self.shift_max, self.shift_max)
        x = torch.roll(x, shifts = shift, dims = 1)
        
        return x


# Create custom dataset (subclassing torch.utils.Dataset)
class TESSDataset(Dataset):
    def __init__(self, labels_df, lightcurves_dir, transform = None, return_id = False, aux_cols = None, aux_stats = None):
        """
        Args: 
        labels_df (pandas.DataFrame): The table containing TIC_IDs and Labels
        lightcurves_dir (Path): The folder containing .npy files
        """
        
        self.labels_df = labels_df
        self.lightcurves_dir = Path(lightcurves_dir)
        self.transform = transform
        self.return_id = return_id
        self.aux_cols = aux_cols
        self.aux_stats = aux_stats

        valid_rows = []
        for _, row in labels_df.iterrows():
            if (self.lightcurves_dir / f"TIC_{int(row['TIC_ID'])}.npy").is_file():
                valid_rows.append(row)
        
        self.labels_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"Dataset initialized with {len(self.labels_df)} strictly synced samples.")

    def __len__(self):
        # Returns the total number of stars
        return len(self.labels_df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find the target star
        row = self.labels_df.iloc[index]
        tic_id = int(row['TIC_ID'])
        label = row['final_label']

        file_path = self.lightcurves_dir / f"TIC_{tic_id}.npy"

        # Handling missing file errors
        #if not file_path.is_file():
        #	flux_arr = np.ones(20000, dtype = np.float32)
        #else:
        flux_arr = np.load(file_path)

        X = torch.tensor(flux_arr, dtype = torch.float32).unsqueeze(dim = 0)

        if self.transform:
            X = self.transform(X)

        y = torch.tensor(label, dtype = torch.float32)

        if self.aux_cols:
            vals = []
            for col in self.aux_cols:
                v = row[col]
                if self.aux_stats:
                    mean, std = self.aux_stats[col]
                    v = (v - mean) / (std + 1e-8)
                vals.append(v)
            aux = torch.tensor(vals, dtype=torch.float32)
            return (X, y, aux, tic_id) if self.return_id else (X, y, aux)

        return (X, y, tic_id) if self.return_id else (X, y)


# Build the custom DataLoaders
def create_dataloaders(csv_path: str, data_dir: str, batch_size: int = 32, aux_cols = None, seed = 42):
    """
    Takes in file paths and resturns train and test dataloaders
    """

    # Load the labels file
    df = pd.read_csv(csv_path)
    df = df[
        (df["Period (days)"] > 0) &
        (df["Duration_hrs"] > 0) &
        (df["Depth_ppm"] > -1)
    ].reset_index(drop=True)

    df["log_period"] = np.log(df["Period (days)"])
    df["log_depth"] = np.log1p(df["Depth_ppm"])

    df["duration_fraction"] = df["Duration_hrs"] / (df["Period (days)"] * 24.0)
    df["log_duration_fraction"] = np.log(df["duration_fraction"])

    check_cols = list(set((aux_cols or []) + ["log_period", "log_depth"]))
    df = df[np.isfinite(df[check_cols]).all(axis=1)].reset_index(drop=True)
    available_files = {Path(f).stem for f in os.listdir(data_dir) if f.endswith('.npy')}
    
    def file_exists(tic_id):
            str_id = str(int(tic_id))
            return str_id in available_files or f"TIC_{str_id}" in available_files

       # Filter out missing files (like isolated eclipsing binaries) and reset index
    df = df[df['TIC_ID'].apply(file_exists)].reset_index(drop=True)
    
    # Split data into training(0.8) and testing(0.2) sets
    if len(df) < 5:
        train_df = df.copy()
        test_df = df.copy()
    else:
        train_df, test_df = train_test_split(
            df,
            test_size = 0.2,
            random_state = 42,
            stratify = df['final_label']
        )
    # NOTE: It is important to use stratify since most of the data will yield result = 0 (no planet) and hence both training and testing splits should get the same ratio of 1:0 results

    # Reset indices since train_test_split shuffles them
    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)

    aux_stats = None
    if aux_cols:
        aux_stats = {col: (train_df[col].mean(), train_df[col].std()) for col in aux_cols}

    # Create Datasets
    train_data = TESSDataset(labels_df = train_df, lightcurves_dir = data_dir, transform = TESSTransform(seed = seed), aux_cols = aux_cols, aux_stats = aux_stats)
    test_data = TESSDataset(labels_df = test_df, lightcurves_dir = data_dir, transform = None, aux_cols = aux_cols, aux_stats = aux_stats)

    g = torch.Generator()
    g.manual_seed(seed)

    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        shuffle = True,
        generator = g
    )

    test_dataloader = DataLoader(
        dataset = test_data,
        batch_size = batch_size,
        shuffle = False
    )

    return train_dataloader, test_dataloader

def get_clean_splits(csv_path, data_dir, aux_cols=None, seed=42):
    df = pd.read_csv(csv_path)
    df = df[(df["Period (days)"] > 0) & (df["Duration_hrs"] > 0) & (df["Depth_ppm"] > -1)].reset_index(drop=True)
    df["log_period"] = np.log(df["Period (days)"])
    df["log_depth"] = np.log1p(df["Depth_ppm"])

    check_cols = list(set((aux_cols or []) + ["log_period", "log_depth"]))
    df = df[np.isfinite(df[check_cols]).all(axis=1)].reset_index(drop=True)

    available_files = {Path(f).stem for f in os.listdir(data_dir) if f.endswith('.npy')}
    def file_exists(tic_id):
        str_id = str(int(tic_id))
        return str_id in available_files or f"TIC_{str_id}" in available_files
    df = df[df['TIC_ID'].apply(file_exists)].reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df['final_label'])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def make_dataloader(df, data_dir, batch_size, aux_cols, aux_stats, transform, shuffle, seed):
    dataset = TESSDataset(labels_df=df, lightcurves_dir=data_dir, transform=transform,
                           aux_cols=aux_cols, aux_stats=aux_stats)
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)