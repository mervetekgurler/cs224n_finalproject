import argparse
import os
from datasets import load_dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, EarlyStoppingCallback
import gc
import torch

def load_and_prepare_datasets(train_file, dev_file):
    train_dataset = load_dataset('csv', data_files=train_file, split='train')
    dev_dataset = load_dataset('csv', data_files=dev_file, split='train')

    datasets = DatasetDict({
        'train': train_dataset,
        'validation': dev_dataset
    })
    return datasets

# After finetun_v1, I got a lot of error about as_target_tokenizer, so I changed the function following
# the same logic as the preprocess function in this code: 
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py

# max_length here has to do with the way that Helsinki NLP model does tokenization
# This kept popping up in the finetuning output
# {'max_length': 512, 'num_beams': 4, 'bad_words_ids': [[57059]], 'forced_eos_token_id': 43741}

def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(examples["turkish"], truncation=True, max_length=512, padding=True)
    labels = tokenizer(text_target=examples["english"], truncation=True, max_length=512, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def tokenize_datasets(datasets, tokenizer):
    tokenized_datasets = datasets.map(lambda examples: tokenize_function(examples, tokenizer), batched=True, remove_columns=["turkish", "english"])
    return tokenized_datasets

# I used the medium post below for compute_metrics function
# https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f

# def compute_metrics(eval_pred):
#     metric = load_metric("sacrebleu")
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them
#     labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # SacreBLEU expects a list of predictions and a list of lists of references
#     decoded_labels = [[label] for label in decoded_labels]
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     return {"bleu": result["score"]}

# https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb

metric = load_metric("sacrebleu")

import numpy as np

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13?page=2

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def main(train_file, dev_file, model_checkpoint, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    datasets = load_and_prepare_datasets(train_file, dev_file)
    tokenized_datasets = tokenize_datasets(datasets, tokenizer)

    from transformers import DataCollatorForSeq2Seq

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_num_workers=1,
        evaluation_strategy="steps",  # Change evaluation strategy to steps for early callback
        eval_steps=1000, 
        save_steps=1000,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=6,
        # this is because finetune_V1 logs showed promise of continous improvement without overfitting
        # ie. the training loss continued to decrease without a corresponding increase in validation loss
        # so I implemented early stopping to stop training if the BLEU score does not improve for 3 evaluations
        # while still increasing epochs to 6
        fp16=True,
        dataloader_prefetch_factor=2,
        #predict_with_generate=True, # this generates text during evaluation which takes more compute
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="bleu",  # Specify the metric for early stopping and best model selection
        greater_is_better=True,  # to match BLEU score higher is better
        gradient_accumulation_steps=2, # for the training to feel like a higher batch size like 8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    # Force a garbage collection to free up unreferenced memory
    gc.collect()

    # Clear the CUDA memory cache
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for translation.")
    parser.add_argument("--train_file", type=str, required=True, help="The path to the training data csv file.")
    parser.add_argument("--dev_file", type=str, required=True, help="The path to the validation data csv file.")
    parser.add_argument("--model_checkpoint", type=str, default="Helsinki-NLP/opus-mt-tc-big-tr-en", help="Model identifier from Huggingface Model Hub.")
    parser.add_argument("--output_dir", type=str, default="./results_v2_BLEU", help="Where to store the fine-tuned model.")
    
    args = parser.parse_args()
    
    main(args.train_file, args.dev_file, args.model_checkpoint, args.output_dir)
