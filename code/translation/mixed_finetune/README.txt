In this section of the repo, we share the code for a mini experiment with Helsinki NLP. We finetuned the Helsinki NLP model using all the data that we have except for the Ottoman manuscript Osman Aga. The training data in this case includes not only Turkish-English novel pairs but also the Ottoman novel and manuscript set aside for testing. We wanted to experiment with a mixed language finetuning approach to see if that would yield better results. Then we tested our model on the Osman Aga manuscript.

Here are the dataset statistics (in number of sentence pairs):
Train Set 44,276
Dev Set 11,070
Test Set Osman Aga 629

Here are the results:
BLEU score: 3.8721796518165257
chrF score: 24.22828727838866
