
# Baselines to be implemented
The following baselines are proposed for the task **the cross-lingual adaptation of hate speech classification for low-resource languages**:

| network architecture | training method | train dataset | test dataset | asignee | extra comments | current status
| --- | --- | --- | --- | --- | --- | --- |
| (mBERT / XLM-R / multilingual embedding+LSTM) + CH | simple | monolingual | monolingual and/or crosslingual | Tanmay | <ul></ul> | <ul><li>only XLM-R implemented for now<li>need to expand to other architectures</ul> |
| (mBERT / XLM-R / multilingual embedding+LSTM) + CH | partially freeze model | mixed (eg- train on English then fine-tune on Spanish) | monolingual and/or crosslingual | |<ul></ul> | |
| (BERTweet) + CH | simple | English only | English and/or Spanish translated to English  | |<ul><li>English evaluation is direct<li> Spanish will need translation to english before evaluating</ul> | |
| [Cross-domain and Cross-lingual Abusive Language Detection: a Hybrid Approach with Deep Learning and a Multilingual Lexicon](https://aclanthology.org/P19-2051.pdf) | (as in paper) | (as in paper) | (as in paper) | | <ul><li>offline machine translation<li>uses hurtlex</ul>| |
| [Rumour Detection via Zero-shot Cross-lingual Transfer Learning](https://arxiv.org/pdf/2109.12773.pdf) | (as in paper) | (as in paper) | (as in paper) | | <ul><li>Student-teacher method</ul>| |
| [Deep Short Text Classification with Knowledge Powered Attention](https://arxiv.org/pdf/1902.08050.pdf), implementation present [here](https://github.com/AIRobotZhang/STCKA) | (as in paper) | | | Tanmay | <ul><li>uses spacy entity extractor, [Microsoft Concept Graph](https://concept.research.microsoft.com/Home/Download) out of the box</ul> | <ul><li>implemented only with english for now due to dependancy on language-specific KGs<li>current findings suggest that better <ul><li>entity extraction<li> concept graphs</ul> are required</ul> |
| [Modelling Latent Translations for Cross-Lingual Transfer](https://arxiv.org/pdf/2107.11353.pdf) | | | | | <ul><li>online machine translation<li>might not be a suitable baseline at the moment</ul>| |

Notes:

1. `CH` means `classification head`.
