# pip install openai
# pip install -q sacrebleu

import os
import sys
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from openai import OpenAI

api_key = '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'
client = OpenAI(api_key=api_key)

def translate_with_openai(sentence, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": "Translate this sentence to English: " + sentence},
        ]
    )
    return response.choices[0].message.content

def main(test_set_path, model, directory_name):
    # Normalize the test set path
    test_set_path = os.path.normpath(test_set_path)

    # Load the test set
    test_set_df = pd.read_csv(test_set_path)

    # Translate the Turkish sentences
    print("Translating sentences...")
    test_set_df['model_translation'] = test_set_df['turkish'].apply(lambda x: translate_with_openai(x, model))

    # Calculate BLEU and chrF scores
    bleu = BLEU()
    chrf = CHRF()
    bleu_score = bleu.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])
    chrf_score = chrf.corpus_score(test_set_df['model_translation'].tolist(), [test_set_df['english'].tolist()])

    # Print and save the results
    results = f"BLEU score: {bleu_score.score}\nchrF score: {chrf_score.score}"
    print(results)

    base_name = os.path.basename(test_set_path).replace("test_", "").rsplit('.', 1)[0]

    # Ensure the directory exists
    os.makedirs(directory_name, exist_ok=True)

    csv_output_file_name = os.path.join(directory_name, f"{model}_{base_name}.csv")
    txt_output_file_name = os.path.join(directory_name, f"{model}_{base_name}_scores.txt")

    # Save the translations alongside the ground truth
    test_set_df.to_csv(csv_output_file_name, index=False)
    print(f"Model outputs saved to {csv_output_file_name}")

    # Save the scores
    with open(txt_output_file_name, 'w') as result_file:
        result_file.write(results)
    print(f"Scores saved to {txt_output_file_name}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <path to test set CSV> <model name> <directory name>")
        sys.exit(1)
    test_set_path = sys.argv[1]
    model_name = sys.argv[2] # Model name either gpt-4-0125-preview or gpt-3.5-turbo-0125
    directory_name = sys.argv[3]  # Directory to save outputs
    main(test_set_path, model_name, directory_name)
