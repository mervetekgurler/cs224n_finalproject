# pip install -q transformers
# pip install -q sacrebleu
import os
import sys
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sacrebleu.metrics import BLEU, CHRF

def aya_translate(text, tokenizer, aya_model):
    tur_inputs = tokenizer.encode("Translate to English:" + text, return_tensors="pt")
    tur_outputs = aya_model.generate(tur_inputs, max_new_tokens=128)

    return tokenizer.decode(tur_outputs[0])  

def main(test_set_path):
    # model
    checkpoint = "CohereForAI/aya-101"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    # Normalize the test set path
    test_set_path = os.path.normpath(test_set_path)

    # Load the test set
    test_set_df = pd.read_csv(test_set_path)

    # Translate the Turkish sentences
    print("Translating sentences...")
    test_set_df['model_translation'] = test_set_df['turkish'].apply(lambda x: aya_translate(x, tokenizer, aya_model))

    # Calculate BLEU and chrF scores
    bleu = BLEU()
    chrf = CHRF()
    bleu_score = bleu.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])
    chrf_score = chrf.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])

    # Print and save the results
    results = f"BLEU score: {bleu_score.score}\nchrF score: {chrf_score.score}"
    print(results)

    base_name = os.path.basename(test_set_path).replace("test_", "").rsplit('.', 1)[0]
    directory_name = "aya_baseline"
    # Ensure the directory exists
    os.makedirs(directory_name, exist_ok=True)

    csv_output_file_name = os.path.join(directory_name, f"aya_{base_name}.csv")
    txt_output_file_name = os.path.join(directory_name, f"aya_{base_name}_scores.txt")

    # Save the translations alongside the ground truth
    test_set_df.to_csv(csv_output_file_name, index=False)
    print(f"Model outputs saved to {csv_output_file_name}")

    # Save the scores
    with open(txt_output_file_name, 'w') as result_file:
        result_file.write(results)
    print(f"Scores saved to {txt_output_file_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python baseline.py <path to test set CSV>")
        sys.exit(1)
    test_set_path = sys.argv[1]
    main(test_set_path)
