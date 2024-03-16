import csv
import glob
import os
import sys

def process_aligned_files(directory_path):
    """
    Process all '.txt.aligned' files in the specified directory,
    converting each to a CSV file with columns ['turkish', 'english', 'similarity_score'].

    Args:
    directory_path (str): The path to the directory containing '.txt.aligned' files.
    """

    # Construct the path pattern to match all '.txt.aligned' files in the directory
    path_pattern = os.path.join(directory_path, '*.txt.aligned')

    # Use glob to find all files that match the pattern
    files = glob.glob(path_pattern)

    # Check if no files are found
    if not files:
        print(f"No '.txt.aligned' files found in {directory_path}.")
        return
    
    # Create output folder
    output_folder = '/Users/mervetekgurler/Desktop/PhD/PhD Classes/Fifth Year/CS 224N/Final Project/SentAlign/translation' + '/raw_corpus'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over each file found
    for file_path in files:
        # Extract the base name for the output CSV file
        base_name = os.path.basename(file_path)
        csv_file_name = base_name.replace('.txt.aligned', '_aligned.csv')
        output_csv_path = os.path.join(output_folder, csv_file_name)
        
        # Open the input .txt.aligned file and the output CSV file
        with open(file_path, 'r', encoding='utf-8') as infile, open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            # Write the header row
            csv_writer.writerow(['turkish', 'english', 'similarity_score'])
            
            for line in infile:
                parts = line.strip().split('\t')
                # Write the parts to the CSV file if the line has exactly three parts
                if len(parts) == 3:
                    csv_writer.writerow(parts)
        
        print(f'Data from {file_path} compiled and saved to {output_csv_path}')

if __name__ == '__main__':
    # Check if the user has provided a directory path as an argument
    if len(sys.argv) != 2:
        print("Usage: python process_aligned_texts.py <directory_path>")
        sys.exit(1)

    # Get the directory path from the command line arguments
    directory_path = sys.argv[1]

    # Process the .txt.aligned files in the specified directory
    process_aligned_files(directory_path)
