# -*- coding: utf-8 -*-

#import tagme
import logging
import sys
import os.path
import requests
import json
from tqdm import tqdm

# 标注的“Authorization Token”，需要注册才有
#tagme.GCUBE_TOKEN = "d866f962-a8f3-4213-a93b-fc0c1383a973-843339462"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

def get_instance_concept(file):
    ent_concept = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            cpt = line[0]
            ent = line[1]
            if ent not in ent_concept:
                ent_concept[ent] = []
            ent_concept[ent].append(cpt)

    return ent_concept

def Annotation_mentions(txt):
    """
    发现那些文本中可以是维基概念实体的概念
    :param txt: 一段文本对象，str类型
    :return: 键值对，键为本文当中原有的实体概念，值为该概念作为维基概念的概念大小，那些属于维基概念但是存在歧义现象的也包含其内
    """
    annotation_mentions = tagme.mentions(txt)
    dic = dict()
    for mention in annotation_mentions.mentions:
        try:
            dic[str(mention).split(" [")[0]] = str(mention).split("] lp=")[1]
        except:
            logger.error('error annotation_mention about ' + mention)
    return dic

def get_tagme_mentions(txt):

    annotation_mentions = tagme.mentions(txt)
    dic = dict()
    ret_mentions = []
    for mention in annotation_mentions.mentions:
        try:
            ret_mentions.append(str(mention).split(" [")[0].lower())
        except:
            logger.error('error annotation_mention about ' + mention)
    return ret_mentions


def Annotate(txt, language="en", theta=0.1):
    """
    解决文本的概念实体与维基百科概念之间的映射问题
    :param txt: 一段文本对象，str类型
    :param language: 使用的语言 “de”为德语, “en”为英语，“it”为意语.默认为英语“en”
    :param theta:阈值[0, 1]，选择标注得分，阈值越大筛选出来的映射就越可靠，默认为0.1
    :return:键值对[(A, B):score]  A为文本当中的概念实体，B为维基概念实体，score为其得分
    """
    annotations = tagme.annotate(txt, lang=language)
    dic = dict()
    try:
        for ann in annotations.get_annotations(theta):
            # print(ann)
            try:
                A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
                dic[(A, B)] = score
            except:
                logger.error('error annotation about ' + ann)
    except:
        pass
    return dic


if __name__ == '__main__':
    import spacy
    import time
    import pandas as pd
    df = pd.read_pickle('dataset/hateval2019en_train.pkl')#.iloc[:10]
    file = 'dataset/data-concept-instance-relations.txt'
    k = 5
    nlp = spacy.load("en_core_web_sm")
    ent_concept = get_instance_concept(file)
    concept_num, concept_den = 0,0
    with open('dataset/hateval2019en_train.tsv', 'wt', encoding='utf-8') as f_w:
        for index,row in df[["text","label"]].iterrows():
    #with open('hateval2019en.txt', 'rt', encoding='utf-8') as f, open('hateval2019en.tsv', 'wt', encoding='utf-8') as f_w:
        #for ii, line in enumerate(f):
            #text,label = line.strip().split('\t')
            text,label = row.text.replace('\n',' ').replace('\t',' ').strip(),row.label
            if text == '': text = 'no text'
            if label == '': raise ValueError()
            print('text =',text)
            print('label =',label)

            doc = nlp(text)
            obj = []

            for ent in doc.ents:
                print('entity extracted =',ent.text)
                obj.append(ent.text.lower())
#           obj = get_tagme_mentions(text)

            concept = []
            for ent in obj:
                print('looping for entity:',ent)
                if ent in ent_concept:
                    print('entity found in KG!')
                    length = len(ent_concept[ent])
                    length = k if length > k else length
                    concept.extend(ent_concept[ent][0:length])
                    print('concepts found:',','.join(ent_concept[ent][0:length]))
                else:
                    print('concept NOT found in KG! simply adding',ent)
                    #concept.append(ent)
            concept_num += len(concept)
            concept_den += 1
            if len(concept) == 0:
                print('No concept found, appending `None` concept')
                concept.append('None')

            f_w.write(text+'\t'+' '.join(concept)+'\t'+str(label)+'\n')
    print('concept_num = ',concept_num)
    print('concept_den = ',concept_den)
    print('concepts per sentence = ',concept_num/concept_den)
