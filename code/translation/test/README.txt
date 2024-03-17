We evaluated the fine-tuned models by using the best model and by testing the performance on all 3 checkpoints. Our goal is to further investigate the differences between BLEU and chr-F scores and how they contribute to translation output and whether there are differences between a model that is great for the novel but not the manuscript tests.
We have a hunch that the differences in translation practices with regards to person and placenames could be causing lower BLEU scores, when even if the model translation is not wrong. For example, Ottoman Temeşvar in modern-day Romania can be translated as Timișoara using its current name or as Temesvár using its historical Hungarian name or just retained as Temeşvar. We should account for these complexities when evaluating historical translation. 

We obtained the following results from the best models:

finetune_v1
manuscript
BLEU score: 3.288213268485018
chrF score: 23.240166897569235

novel
BLEU score: 10.939317408733729
chrF score: 33.519459163707566

osmanaga
BLEU score: 2.776063497445394
chrF score: 20.068630988005225

finetune_v2_BLEU
manuscript
BLEU score: 3.313619990622139
chrF score: 23.470219100411015

novel
BLEU score: 10.623811620573633
chrF score: 33.070025214491075

osmanaga
BLEU score: 2.8486204889879794
chrF score: 20.157962485064314
