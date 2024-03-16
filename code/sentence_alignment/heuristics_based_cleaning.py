from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import glob
import os
import sys

def clean_csv(file_path, output_directory, n_clusters=10):
    """
    Clean a CSV file based on dynamically calculated similarity score cutoff
    from k-means clustering and sentences being at least 2 letters long,
    then save the cleaned data to a new directory with modified file names.

    Args:
    file_path (str): Path to the CSV file.
    output_directory (str): Path to the directory where cleaned files will be saved.
    n_clusters (int): Number of clusters to use in k-means for determining the cutoff.
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Ensure there are enough data points for clustering
    if len(df) > n_clusters:
        similarity_scores = df['similarity_score'].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(similarity_scores)
        cutoff = sorted(kmeans.cluster_centers_.flatten())[2]
    else:
        cutoff = 0.563

    df_filtered = df[(df['turkish'].str.len() >= 2) & (df['english'].str.len() >= 2)]
    df_filtered = df_filtered[df_filtered['similarity_score'] >= cutoff]

    # Construct the new output file path
    base_name = os.path.basename(file_path)
    new_file_name = base_name.replace('_aligned.csv', '_final.csv')
    output_csv_path = os.path.join(output_directory, new_file_name)

    # Save the cleaned data
    df_filtered.to_csv(output_csv_path, index=False)

def main(directory_path, output_directory):
    """
    Process all CSV files in the given directory and save cleaned versions to a new directory.

    Args:
    directory_path (str): The path to the directory containing CSV files.
    output_directory (str): The path to the directory where cleaned files will be saved.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    path_pattern = os.path.join(directory_path, '*.csv')
    csv_files = glob.glob(path_pattern)

    for csv_file in csv_files:
        print(f"Cleaning {csv_file}...")
        clean_csv(csv_file, output_directory)
        print(f"Finished cleaning. Cleaned data saved to {output_directory}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python heuristics_based_cleaning.py <directory_path> <output_directory>")
        sys.exit(1)

    directory_path = sys.argv[1]
    output_directory = sys.argv[2]
    main(directory_path, output_directory)
