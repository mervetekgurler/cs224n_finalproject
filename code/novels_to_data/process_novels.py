"""
process_novels.py
Author: Merve Tekgürler
Date: 03/06/2024

Description:
    This script automates the processing of all EPUB and PDF files in a specified directory ('novels_file') by
    calling the text_extractor script for each file. It determines the language of each file based on the
    presence of '_en' or '_tr' in the filenames and passes this information along with the file path to
    'text_extractor.py'. This allows for batch processing of files for text extraction and sentence tokenization.

Usage:
    python process_novels.py

Notes:
    Ensure that text_extractor.py is in the same directory as this script or adjust the 'script_name' variable
    to point to the correct location of text_extractor.py. The directory to be processed is hardcoded as
    'novels_file'; modify this variable if a different directory name is used.

"""

import subprocess
import os

def determine_language(filename):
    if '_tr' in filename:
        return 'turkish'
    elif '_en' in filename:
        return 'english'
    else:
        raise ValueError("Filename does not specify a language with '_en' or '_tr'.")

def main():
    novels_dir = 'novels_file'
    script_name = 'text_extractor.py'  

    for filename in os.listdir(novels_dir):
        file_path = os.path.join(novels_dir, filename)
        if not os.path.isfile(file_path):
            continue
        
        try:
            language = determine_language(filename)
        except ValueError as e:
            print(f"Skipping {filename}: {e}")
            continue

        # Build the command to call the processing script
        command = ['python', script_name, file_path, language]
        
        try:
            # Execute the command
            subprocess.run(command, check=True)
            print(f"Processed {filename} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {filename}: {e}")

if __name__ == "__main__":
    main()
