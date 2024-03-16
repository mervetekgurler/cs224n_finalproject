To get baselines in the machine translation task we used Google Gemini, OpenAI GPT-3.5 and GPT-4, Cohere Aya, and Helsinki NLP's model before fine-tuning.

Helsinki NLP
manuscript
BLEU score: 3.4387141311362677
chrF score: 22.20585076369852

novel
BLEU score: 9.740732160707617
chrF score: 33.253324973488056

osmanaga
BLEU score: 2.8303861534363963
chrF score: 19.391157810126288

Cost: GCP Free Credits (est. $4-5)

Gemini
manuscript
All translations - BLEU score: 8.86924897664771, chrF score: 36.546398551524575 
Filtered (valid translations only) - BLEU score: 9.057007633205238, chrF score: 39.55344621266212

novel
All translations - BLEU score: 10.831756727112259, chrF score: 34.69801347643916 
Filtered (valid translations only) - BLEU score: 10.967800690810428, chrF score: 37.25066779332506

osmanaga
All translations - BLEU score: 7.014876103281617, chrF score: 32.07380828704673
Filtered (non-empty and valid translations) - BLEU score: 7.848347281490299, chrF score: 36.61478502594365

Cost: Free

Gemini (all safety settings closed)
manuscript_no_safety
All translations - BLEU score: 9.040167708707436, chrF score: 38.9620815453946 
Filtered (valid translations only) - BLEU score: 9.040167708707436, chrF score: 39.044613686393944

novel_no_safety
All translations - BLEU score: 11.10913027990284, chrF score: 37.32569185362411 
Filtered (valid translations only) - BLEU score: 11.10913027990284, chrF score: 37.37718483075166

osmanaga_no_safety
All translations - BLEU score: 7.8421801129838675, chrF score: 36.40565274349136
Filtered (non-empty and valid translations) - BLEU score: 7.8421801129838675, chrF score: 36.59685268728821

Cost: Free

GPT-3.5
manuscript
BLEU score: 8.230243544012923
chrF score: 38.58055391366726

novel
BLEU score: 11.143521636182633
chrF score: 38.09210331259963

osmanaga
BLEU score: 7.105613238002907
chrF score: 35.84064769623547

Cost: $0.25

GPT-4
manuscript
BLEU score: 9.751330692584272
chrF score: 41.4068544109814

novel
BLEU score: 11.677913704959238
chrF score: 39.47366754150075

osmanaga
BLEU score: 7.974807752759638
chrF score: 37.714786665604514

Cost: $4.18

Cohere Aya
manuscript
BLEU score: 5.049970582628528
chrF score: 29.487182826608805

novel
BLEU score: 7.296032830144639
chrF score: 33.384249345753005

osmanaga


Cost: GCP Free Credits (est. $10)

