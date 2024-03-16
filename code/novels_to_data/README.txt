In this part of the repo, you can find the code and approaches we used to turn novels with translations into sentence level parallel datasets.
The pipeline is as follows:
1. Acquire epubs and PDFs of novels through institutional and other library access
2. Extract the texts and divide them up into sentences
  2.1 PDFs are OCR'ed on AbbyyFineReader 15 into txt files
  2.2 EPUBs are extracted using ebooklib and BeautifulSoup and in some cases https://convertio.co/epub-txt/
3. Turn the sentence files into the format expected by SentAlign (https://github.com/steinst/SentAlign)
