In this part of the repo, you can find the code we used to fine-tune the Helsinki NLP Opus MT Turkish-English translation model: https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-tr-en
We used the Hugging Face Trainer class and followed these tutorials and code bases:
1. https://huggingface.co/docs/transformers/en/training
2. https://github.com/huggingface/notebooks/blob/main/examples/translation.ipynb
3. https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f
4. https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py

Here are the statistics of the datasets (in number of sentence pairs):
Train Set 41,782
Dev Set 10,447
Test Set 1: Novel 2,694
Test Set 2: Manuscript 425
