"""
text_extractor.py
Author: Merve Tekgürler
Date: 03/06/2024

Description:
    This script processes individual EPUB or PDF files to extract text and then tokenize the text into sentences.
    It handles EPUB files by extracting text directly and PDF files by reading previously OCR-processed text files.
    For EPUB files, the script saves the extracted full text in a 'novels_extracted' directory and the tokenized sentences 
    in a 'novels_sentences/english' or 'novels_sentences/turkish' directory based on the specified language.
    
    For PDF files, this script identifies OCR'ed text files and extract sentences in the same way as for EPUBs. This
    script expects OCR'ed files in novels_extracted folder in the same directory as the novels_file. Please use the
    appropriate OCR software. I recommend ABBYY FineReader 15, Windows version.
    
    The sentence extraction matches the input requirements of the SentAlign repo (https://github.com/steinst/SentAlign)

Notes:
    This script assumes that the files are named in a very specific way. Below are the examples:
    PDF: yasarkemal_memed_en.pdf -> novels_extracted/yasarkemal_memed_en.txt
    EPUB: tanpinar_huzur_en.epub
    The _en or _tr at the end helps determine the language of the text and the output folder for the extracted files.
    Everything leading up to the en is optional but is has to be the same in English and in Turkish for SentAlign
    
Usage:
    python text_extractor.py <file_path> <language>

Arguments:
    file_path: The path to the EPUB or PDF file to be processed.
    language: The language of the text ('english' or 'turkish'), used for sentence tokenization.

Dependencies:
    - Python 3.x
    - NLTK
    - ebooklib
    - BeautifulSoup4
"""

import argparse
import os
from ebooklib import epub
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have the necessary NLTK data (punkt tokenizer models)
nltk.download('punkt', quiet=True)

def get_text_from_pdf(pdf_path):
    # Assuming OCR'ed text file has the same basename as the PDF file but with a .txt extension
    text_path = f"novels_extracted/{os.path.basename(pdf_path).rsplit('.', 1)[0]}.txt"
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    items = list(book.get_items())

    text_blocks = []
    for item in items:
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        # Remove in-line footnote references
        for footnote_ref in soup.find_all('a', class_='si-noteref'):
            footnote_ref.decompose()
        # Remove the entire footnotes section
        for footnotes_section in soup.find_all('div', class_='si-footnotes'):
            footnotes_section.decompose()

        for paragraph in soup.find_all('p'):
            paragraph_text = paragraph.get_text(separator=' ', strip=True)
            if paragraph_text:
                text_blocks.append(paragraph_text)
    return '\n\n'.join(text_blocks)

def save_texts(full_text, sentences, base_path, language, original_base_path):
    # Create the directory for extracted full texts
    os.makedirs('novels_extracted', exist_ok=True)
    full_text_path = f"novels_extracted/{original_base_path}.txt"
    
    # Only write the full text if the file does not already exist
    if not os.path.exists(full_text_path):
        with open(full_text_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
    else:
        print(f"Skipping full text save for '{original_base_path}'; file already exists.")

    # Language folder based on full language name
    sentences_dir = f"novels_sentences/{language}"
    os.makedirs(sentences_dir, exist_ok=True)

    # For sentence files, we use the base_path without language code
    sentences_path = f"{sentences_dir}/{base_path}.txt"
    
    # Only write the sentences if the file does not already exist
    if not os.path.exists(sentences_path):
        with open(sentences_path, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
    else:
        print(f"Skipping sentence save for '{base_path}'; file already exists.")

def main():
    parser = argparse.ArgumentParser(description="Extract text from an EPUB or PDF file.")
    parser.add_argument('file_path', type=str, help='The path to the EPUB or PDF file')
    parser.add_argument('language', choices=['english', 'turkish'], help='The language for sentence tokenization ("english" or "turkish")')
    args = parser.parse_args()

    # Determine the file type
    file_type = args.file_path.rsplit('.', 1)[-1].lower()

    if file_type == 'epub':
        full_text = get_text_from_epub(args.file_path)
    elif file_type == 'pdf':
        full_text = get_text_from_pdf(args.file_path)
    else:
        raise ValueError("Unsupported file type. Please provide an EPUB or PDF file.")

    # Tokenize the full text into sentences
    sentences = sent_tokenize(full_text, language=args.language.lower())

    # Derive base filename without the extension
    original_base_path = os.path.basename(args.file_path).rsplit('.', 1)[0]
    base_path = original_base_path

    # Adjust base_path for sentences by removing language code
    if args.language.lower() == 'turkish' and base_path.endswith('_tr'):
        base_path = base_path[:-3]
    elif args.language.lower() == 'english' and base_path.endswith('_en'):
        base_path = base_path[:-3]

    # Save the extracted texts
    save_texts(full_text, sentences, base_path, args.language.lower(), original_base_path)

if __name__ == "__main__":
    main()
