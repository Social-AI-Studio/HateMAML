
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
| Dataset | Lang | Train | Test| Dev | Domain | F1 |
| --- | --- | --- | --- | --- | --- | --- |
|HateEval19| |  
|LCS2 | |
|SemEval20| |  

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