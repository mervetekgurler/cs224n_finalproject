import os
import sys
import re
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU, CHRF

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_checkpoint(test_set_df, checkpoint_path, tokenizer, base_name, directory_name):
    # Load the model from the checkpoint
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    # Translate the Turkish sentences
    print(f"Translating sentences using {checkpoint_path}...")
    test_set_df['model_translation'] = test_set_df['turkish'].apply(lambda x: translate(x, model, tokenizer))

    # Calculate BLEU and chrF scores
    bleu = BLEU()
    chrf = CHRF()
    bleu_score = bleu.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])
    chrf_score = chrf.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])

    # Print and save the results
    results = f"BLEU score: {bleu_score.score}\nchrF score: {chrf_score.score}"
    print(results)

    checkpoint_name = os.path.basename(checkpoint_path)
    csv_output_file_name = os.path.join(directory_name, f"{base_name}_{checkpoint_name}.csv")
    txt_output_file_name = os.path.join(directory_name, f"{base_name}_{checkpoint_name}_scores.txt")

    # Save the translations alongside the ground truth
    test_set_df.to_csv(csv_output_file_name, index=False)
    print(f"Model outputs saved to {csv_output_file_name}")

    # Save the scores
    with open(txt_output_file_name, 'w') as result_file:
        result_file.write(results)
    print(f"Scores saved to {txt_output_file_name}")

def main(test_set_path, results_directory="results"):
    # Normalize the test set path
    test_set_path = os.path.normpath(test_set_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-tr-en")

    # Load the test set
    test_set_df = pd.read_csv(test_set_path)

    base_name = os.path.basename(test_set_path).replace("test_", "").rsplit('.', 1)[0]
    directory_name = "helsinki_finetuned_test_all_V2_BLEU"
    # Ensure the directory exists
    os.makedirs(directory_name, exist_ok=True)

    # Automatically discover all checkpoints in the results directory
    checkpoints = [os.path.join(results_directory, d) for d in os.listdir(results_directory) if re.match(r'checkpoint-\d+', d)]

    for checkpoint in checkpoints:
        evaluate_checkpoint(test_set_df, checkpoint, tokenizer, base_name, directory_name)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <path to test set CSV> <path to results directory>")
        sys.exit(1)
    test_set_path = sys.argv[1]
    results_directory = sys.argv[2]
    main(test_set_path, results_directory)
