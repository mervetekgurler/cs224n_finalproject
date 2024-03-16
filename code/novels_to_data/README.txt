In this part of the repo, you can find the code and approaches we used to turn novels with translations into sentence level parallel datasets.
The pipeline is as follows:
1. Acquire epubs and PDFs of novels through institutional and other library access
2. Extract the texts 
  2.1. PDFs are OCR'ed on AbbyyFineReader 15 into txt files
  2.2. EPUBs are extracted using ebooklib and BeautifulSoup and in some cases https://convertio.co/epub-txt/
3. Divide the texts into sentences
4. Map the sentence txts into the folder structure expected by SentAlign: https://github.com/steinst/SentAlign
  The folder structure of source and target language texts is this outlined in the SentAlign repo and it looks like this in my case:
  /path/to/files/turkish/file1.txt
  /path/to/files/turkish/file2.txt
  ...
  /path/to/files/english/file1.txt
  /path/to/files/english/file2.txt
5. Align the texts using SentAlign on GCP
  This process requires both a GPU for embedding the sentences and overlaps of sentences with LaBSE embeddings (https://huggingface.co/sentence-transformers/LaBSE), unless one manually edits the requirement for GPU and inserts an argument for CPU. It also requires a lot of RAM because the alignment algorithm itself is rather computationally complex. It took on average 1,5 hours per novel but some as long as 2+ hours.
6. Clean parallel sentences
  6.1. Evaluate the output:
  Sentence alignment does not mean one-to-one alignment only. Sentences can be matched one-to-many, many-to-one, many-to-many, or some sentences can be omitted during the process if they were insertion into the translation, meaning they do not have a source sentence match or not translated originally, meaning that the source sentence does not have a target translation. This means that the alignment output does not have the same number of sentences as the input. For instance, one novel that contained 8881 source and 9830 target sentences ended up with 7728 sentence pairs.
  SentAlign algorithm calculates similarity scores between sentences during alignment. We kept the default setting that does not accept any similarity score lower than 0.4. This setup allows for higher quality matches to be retained. However, we still had to evaluate the output and develop a heuristic to clean the data further. We calculated distributions of similarity scores and other statistical metrics and read some random subset of the output closely to develop a heuristic 
  6.2. Heuristic for cleaning the dataset:
  For each novel, we clustered the sentences into 10 clusters using k-means clustering on similarity scores and removed those sentence pairs that have a lower similarity score than that of the the 3rd lowest ranked centroid. We applied this approach dynamically to all novels.
7. Create training/evaluation and test sets
  7.1. Train/dev: We combined all the sentence pairs, removed the similarity scores, and shuffled them. Afterwards we split the dataset 80/20 into train and dev sets
  7.2 Test: We selected one novel and one Ottoman manuscript to be our test sets. We set them aside and started acquiring baselines for them.
  7.3 Ottoman train/dev: We selected one longer manuscript (Osman Aga) to be our train/dev set in the second fine-tuning phase. We also got baselines on this manuscript.
