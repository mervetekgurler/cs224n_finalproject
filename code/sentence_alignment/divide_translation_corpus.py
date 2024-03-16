from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
import glob
import os
import sys

def merge_shuffle_split_save(directory_path, output_path, test_size=0.2, sample_size=0.1):
    """
    Merge, shuffle, split into train/dev sets, and create a publication sample.

    Args:
    directory_path (str): Directory containing cleaned CSV files.
    output_path (str): Base directory for outputting the split and sample datasets.
    test_size (float): Proportion of the dataset to include in the dev set.
    sample_size (float): Proportion of the training set to sample for publication.
    """
    # Pattern to match cleaned CSV files
    path_pattern = os.path.join(directory_path, '*_final.csv')
    csv_files = glob.glob(path_pattern)
    all_data = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, usecols=['turkish', 'english'])
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    shuffled_df = shuffle(combined_df, random_state=42)

    # Split into train and dev sets
    train_df, dev_df = train_test_split(shuffled_df, test_size=test_size, random_state=42)

    # Create a publication sample from the training set
    publication_sample_df = train_df.sample(frac=sample_size, random_state=42)

    # Define output file paths
    train_file_path = os.path.join(output_path, 'train.csv')
    dev_file_path = os.path.join(output_path, 'dev.csv')
    sample_file_path = os.path.join(output_path, 'publication_sample.csv')

    # Save datasets to CSV files
    train_df.to_csv(train_file_path, index=False)
    dev_df.to_csv(dev_file_path, index=False)
    publication_sample_df.to_csv(sample_file_path, index=False)

    print(f"Train set saved to {train_file_path}")
    print(f"Dev set saved to {dev_file_path}")
    print(f"Publication sample saved to {sample_file_path}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <directory_path> <output_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    output_path = sys.argv[2]

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    merge_shuffle_split_save(directory_path, output_path)
