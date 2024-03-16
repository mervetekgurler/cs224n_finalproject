import os
import sys
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU, CHRF

# max_length here has to do with the way that Helsinki NLP model does tokenization
# This kept popping up in the finetuning output
# {'max_length': 512, 'num_beams': 4, 'bad_words_ids': [[57059]], 'forced_eos_token_id': 43741}

def translate(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(test_set_df, model_path, tokenizer, results_directory):
    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Translate the sentences
    print(f"Translating sentences using model from {model_path}...")
    test_set_df['model_translation'] = test_set_df['turkish'].apply(lambda x: translate(x, model, tokenizer))

    # Calculate BLEU and chrF scores
    bleu = BLEU()
    chrf = CHRF()
    bleu_score = bleu.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])
    chrf_score = chrf.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])

    # Print and save the results
    results = f"BLEU score: {bleu_score.score}\nchrF score: {chrf_score.score}"
    print(results)

    base_name = os.path.basename(test_set_path).replace("test_", "").rsplit('.', 1)[0]
    csv_output_file_name = os.path.join(results_directory, f"{base_name}_translations.csv")
    txt_output_file_name = os.path.join(results_directory, f"{base_name}_scores.txt")

    # Save the translations alongside the ground truth
    test_set_df.to_csv(csv_output_file_name, index=False)
    print(f"Model outputs saved to {csv_output_file_name}")

    # Save the scores
    with open(txt_output_file_name, 'w') as result_file:
        result_file.write(results)
    print(f"Scores saved to {txt_output_file_name}")

def main(test_set_path, model_path, results_directory, tokenizer_model="Helsinki-NLP/opus-mt-tc-big-tr-en"):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Load the test set
    test_set_df = pd.read_csv(test_set_path)

    # Ensure the results directory exists
    os.makedirs(results_directory, exist_ok=True)

    # Evaluate the best model
    evaluate_model(test_set_df, model_path, tokenizer, results_directory)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <path to test set CSV> <path to best model directory> <results directory>")
        sys.exit(1)
    test_set_path = sys.argv[1]
    model_path = sys.argv[2]
    results_directory = sys.argv[3]
    main(test_set_path, model_path, results_directory)
