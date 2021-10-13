
# Baselines to be implemented
The following baselines are proposed for the task **the cross-lingual adaptation of hate speech classification for low-resource languages**:

| network architecture | training method | train dataset | test dataset | asignee | extra comments | current status
| --- | --- | --- | --- | --- | --- | --- |
| (mBERT / XLM-R / multilingual embedding+LSTM) + CH | simple | monolingual | monolingual and/or crosslingual | Tanmay | <ul></ul> | <ul><li>only XLM-R implemented for now<li>need to expand to other architectures</ul> |
| (mBERT / XLM-R / multilingual embedding+LSTM) + CH | partially freeze model | mixed (eg- train on English then fine-tune on Spanish) | crosslingual (eg- Spanish test split) | Tanmay |<ul><li>as this is a few-shot technique, according to [this](https://aclanthology.org/2021.acl-long.447.pdf) paper it is important to use a standard few-shot sample of the original dataset. </ul> | |
| (BERTweet) + CH | simple | English only | English and/or Spanish translated to English  | Tanmay |<ul><li>English evaluation is direct<li> Spanish will need translation to english before evaluating</ul> | |
| [Cross-domain and Cross-lingual Abusive Language Detection: a Hybrid Approach with Deep Learning and a Multilingual Lexicon](https://aclanthology.org/P19-2051.pdf) | (as in paper) | (as in paper) | (as in paper) | Eshaan | <ul><li>offline machine translation<li>uses hurtlex</ul>| |
| [Rumour Detection via Zero-shot Cross-lingual Transfer Learning](https://arxiv.org/pdf/2109.12773.pdf) | (as in paper) | (as in paper) | (as in paper) | Rabiul | <ul><li>Student-teacher method</ul>| |
| [Deep Short Text Classification with Knowledge Powered Attention](https://arxiv.org/pdf/1902.08050.pdf), implementation present [here](https://github.com/AIRobotZhang/STCKA) | (as in paper) | | | Tanmay | <ul><li>does not talk about the cross-lingual setting<li>**might not be a suitable baseline at the moment** <li>uses <ul><li>spacy entity extractor<li>[Microsoft Concept Graph](https://concept.research.microsoft.com/Home/Download)</ul> out of the box</ul> | <ul><li>implemented only with english for now due to dependancy on language-specific KGs<li>current findings suggest that better <ul><li>entity extraction<li> concept graphs</ul> are required<li>**on hold for now ⚠️**</ul> |
| [Modelling Latent Translations for Cross-Lingual Transfer](https://arxiv.org/pdf/2107.11353.pdf) | | | | | <ul><li>online machine translation<li>**might not be a suitable baseline at the moment**</ul>| <ul><li>**on hold for now ⚠️**</ul> |

Notes:

1. `CH` means `classification head`.


# Dataset Summary
| Dataset | Lang | Train | Test| Dev | Domain | Best F1 | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
|HateEval19| English | 9,000  | 2,971 | | immigrant, woman |
|HateEval19| Spanish | 4,500  | 1,600 | | immigrant, woman |
|LCS2 | | | | | politics, extremism |
|SemEval20| Arabic | 8000 | 200 | | twitter stream | | [pdf](https://arxiv.org/pdf/2006.07235.pdf) |
|SemEval20| Danish | 2961 | 329 |
|SemEval20| Greek | 8743 | 1544 |
|SemEval20| Turkish | 31756 | 3528 |
|HASOC19| English| 5852 | 1153 | | | | [pdf](https://dl.acm.org/doi/10.1145/3368567.3368584)|
|HASOC19| Hindi| 4665 | 1318 | | twitter stream |
|HASOC19| German| 3819 | 850 | 
|HASOC20| English| 3708 | 814 | | | | [pdf](https://dl.acm.org/doi/abs/10.1145/3441501.3441517)|
|HASOC20| Hindi| 2963 | 663 | | 
|HASOC20| German| 2373 | 526 |
|AMI Evalita | English | 4000 | 1000 | | Misogyny | 
|AMI Evalita | Italian | 4000 | 1000 | | Misogyny |
|Ousidhoum19 | English | 5647  | | | fear, hostility, directness|
|Ousidhoum19 | French | 4014  | | | |
|Ousidhoum19 | Arabic | 3353  | | | |
|Waseem16 | English | 11,542 |4,947 | | racism and  sexism | 
|OffensEval19 | English | 13,240 | 860 | | insults, swear, threats |  

# Experimental Results
| Model | Dataset | Lang | Training Setting | Acc | F1 | Notes |  
| --- | --- | --- | --- | --- | --- | --- |
mBERT | HateEvaL | English | | | |
mBERT | HateEvaL | Spanish | Zero-shot|
mBERT | HateEvaL | Spanish | Few-shot|
XLM-R | HateEvaL | English | | | |
XLM-R | HateEvaL | Spanish | Zero-shot|
XLM-R | HateEvaL | Spanish | Few-shot|
mBERT | LCS2 | mixed | | | |
XLM-R | LCS2 | mixed | | | |
mBERT | SemEval20 | Turkish | Zero-shot|
mBERT | SemEval20 | English | | | |
mBERT | SemEval20 | Turkish | Zero-shot|
mBERT | SemEval20 | Danish | Zero-shot|
mBERT | SemEval20 | Arabic | Zero-shot|
mBERT | SemEval20 | Greek | Zero-shot|
mBERT | SemEval20 | Turkish | Few-shot|
mBERT | SemEval20 | Danish | Few-shot|
mBERT | SemEval20 | Arabic | Few-shot|
mBERT | SemEval20 | Greek |  Few-shot|
XLM-R | SemEval20 | English | | | |
XLM-R | SemEval20 | Turkish | Zero-shot|
XLM-R | SemEval20 | Danish | Zero-shot|
XLM-R | SemEval20 | Arabic | Zero-shot|
XLM-R | SemEval20 | Greek | Zero-shot|
XLM-R | SemEval20 | Turkish | Few-shot|
XLM-R | SemEval20 | Danish | Few-shot|
XLM-R | SemEval20 | Arabic | Few-shot|
XLM-R | SemEval20 | Greek | Few-shot|