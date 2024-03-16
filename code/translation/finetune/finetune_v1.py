import argparse
import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

def load_and_prepare_datasets(train_file, dev_file):
    train_dataset = load_dataset('csv', data_files=train_file, split='train')
    dev_dataset = load_dataset('csv', data_files=dev_file, split='train')

    datasets = DatasetDict({
        'train': train_dataset,
        'validation': dev_dataset
    })
    return datasets

def tokenize_datasets(datasets, tokenizer):
    def tokenize_function(examples):
        model_inputs = tokenizer(examples["turkish"], truncation=True)

        # Ensure decoder_input_ids are created for training by shifting the labels to the right
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["english"], truncation=True)
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["turkish", "english"])
    return tokenized_datasets

def main(train_file, dev_file, model_checkpoint, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    datasets = load_and_prepare_datasets(train_file, dev_file)
    tokenized_datasets = tokenize_datasets(datasets, tokenizer)

    from transformers import DataCollatorForSeq2Seq

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_num_workers=1,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        fp16=True,
        dataloader_prefetch_factor=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for translation.")
    parser.add_argument("--train_file", type=str, required=True, help="The path to the training data csv file.")
    parser.add_argument("--dev_file", type=str, required=True, help="The path to the validation data csv file.")
    parser.add_argument("--model_checkpoint", type=str, default="Helsinki-NLP/opus-mt-tc-big-tr-en", help="Model identifier from Huggingface Model Hub.")
    parser.add_argument("--output_dir", type=str, default="./results_v1", help="Where to store the fine-tuned model.")
    
    args = parser.parse_args()
    
    main(args.train_file, args.dev_file, args.model_checkpoint, args.output_dir)
