import json
import logging
import mmap
import os

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import pickle
import requests

logname = 'scripts/logs.log'
logging.basicConfig(filename=logname,
                            filemode='w',
    format="[%(levelname)s] %(asctime)s %(lineno)d %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("es_core_news_sm")  # or spacy.load("es_core_news_md")

PAD_token = 0   # Used for padding short sentences
SOS_token = 1   # Start-of-sentence token
EOS_token = 2   # End-of-sentence token


class ConceptnetEmbedding:
    r"""
        A class that provides utilities for conceptnet embeddings for multilingual setting.
        It contains helper function for creating vocabulary and loading embeddings.
        You should download numberbatch-19.08.txt from https://github.com/commonsense/conceptnet-numberbatch
        and put it inside `data/conceptnet` dir
    """

    def __init__(self):
        self.conceptnet_dir = "data/conceptnet/"
        self.conceptnet_embedding_file = "numberbatch-19.08.txt"
        self.src_data_pkl_dir = os.path.join("data", "raw")

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",
                           SOS_token: "SOS", EOS_token: "EOS"}
        self.vocab_size = 3
        self.num_sentences = 0

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.vocab_size
            self.word2count[word] = 1
            self.index2word[self.vocab_size] = word
            self.vocab_size += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def get_num_lines(self, file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def create_embeddings_dict(self, create_embeeding=False):
        embedding_index_save_path = os.path.join(
            self.conceptnet_dir, 'conceptnet_embeddings_index.pkl')

        if create_embeeding:
            embedding_path = os.path.join(
                self.conceptnet_dir, self.conceptnet_embedding_file)
            embeddings_index = {}
            logger.info("Reading multilingual conceptnet embeddings file {}".format(
                self.conceptnet_embedding_file))
            with open(embedding_path) as f:
                for line in tqdm(f, total=self.get_num_lines(embedding_path)):
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs

            with open(embedding_index_save_path, 'wb')as handle:
                pickle.dump(embeddings_index, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
        else:
            logger.info(
                "Loading embeddings from the cache file at location - {}".format(self.conceptnet_dir))
            with open(embedding_index_save_path, 'rb') as handle:
                embeddings_index = pickle.load(handle)

        logger.info("Loaded {} word vectors".format(len(embeddings_index)))
        return embeddings_index

    def get_embedding_matrix(self, create_embedding_dict=False):
        conceptnet_embedding_matrix_save_path = os.path.join(
            self.conceptnet_dir, 'conceptnet_embedding.npy')
        if create_embedding_dict:
            self.build_vocab_from_hate_corpus()
            embeddings_index = self.create_embeddings_dict(
                create_embeeding=create_embedding_dict)
            # create a weight matrix for words in training docs
            embedding_matrix = np.zeros((self.vocab_size, 300))
            for word, i in tqdm(self.word2index.items(), total=len(self.word2index)):
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

            np.save(conceptnet_embedding_matrix_save_path, embedding_matrix)
        else:
            embedding_matrix = np.load(conceptnet_embedding_matrix_save_path)
        logger.info("Embedding dim {}".format(embedding_matrix.shape))
        return embedding_matrix

    def build_vocab_from_hate_corpus(self):
        files = os.listdir(self.src_data_pkl_dir)
        for idx, file_name in enumerate(files):
            path = os.path.join(self.src_data_pkl_dir, file_name)
            if file_name.endswith(".pkl") and file_name.startswith("hateval2019es"):
                logger.info(
                    "IDX #{} Processing corpus file  # {}".format(idx, path))
                data_df = pd.read_pickle(path, compression=None)
                for text in tqdm(data_df['text'], total=len(data_df['text'])):
                    doc = nlp(text)
                    tokens = [w.text for w in doc]
                    self.add_sentence(tokens)
        logger.info("****** Tokenizer Information *****")
        logger.info("\tTotal sentences {}".format(self.num_sentences))
        logger.info("\tTotal vocabulary {}".format(self.vocab_size))
        logger.info("\tExamples of top word2index # {}".format(
            list(self.word2index.items())[:20]))


class BabelfyHelper:
    def __init__(self) -> None:
        self.key = "64e4838e-8c7a-4d84-9594-4048d8125efc"
        self.src_data_pkl_dir = os.path.join("data", "raw")

    def call_babelfy_api(self, text: str, lang: str):
        babelfy_url_text = f"https://babelfy.io/v1/disambiguate?text={text}&lang={lang}&key={self.key}"
        r = requests.get(babelfy_url_text)
        data = r.json()
        for result in data:
            logger.debug(json.dumps(result, indent=2))
            tokenFragment = result.get('tokenFragment')
            tfStart = tokenFragment.get('start')
            tfEnd = tokenFragment.get('end')
            logger.debug(str(tfStart) + "\t" + str(tfEnd))

            # retrieving char fragment
            charFragment = result.get('charFragment')
            cfStart = charFragment.get('start')
            cfEnd = charFragment.get('end')
            logger.debug(str(cfStart) + "\t" + str(cfEnd))

            # retrieving BabelSynset ID
            synsetId = result.get('babelSynsetID')
            logger.debug(synsetId)

            self.call_synset_api(synsetId, lang="ES")
        
        logger.debug(data)

    def call_synset_api(self, synsetId: str, lang: str):
        synset_url_text = f"https://babelnet.io/v6/getSynset?id={synsetId}&targetLang={lang}&key={self.key}"
        r = requests.get(synset_url_text)
        data = r.json()
        # retrieving BabelSense data
        senses = data['senses']

        all_items = []
        for result in senses:
            # logger.info(json.dumps(result, indent=2))
            properties = result.get("properties")
            # print(json.dumps(properties, indent=2))
            keys = ['lemma']
            item = {key: properties[key] for key in keys}
            all_items.append(item)
        logger.info(json.dumps(all_items, indent=2, ensure_ascii=False))

    def read_sentences(self):
        files = os.listdir(self.src_data_pkl_dir)
        for idx, file_name in enumerate(files):
            idx = 0
            path = os.path.join(self.src_data_pkl_dir, file_name)
            if file_name.endswith(".pkl") and file_name.startswith("hateval2019es"):
                logger.info(
                    "IDX #{} Processing corpus file  # {}".format(idx, path))
                data_df = pd.read_pickle(path, compression=None)
                for text in tqdm(data_df['text'], total=len(data_df['text'])):
                    logger.info("TXT # {}".format(text))
                    self.call_babelfy_api(text, lang="es")
                    idx += 1
                    if idx > 10:
                      break
                    
            if idx > 1:
                break
            

    def json_reader(self):
        sample = [{'tokenFragment': {'start': 1, 'end': 1}, 'charFragment': {'start': 15, 'end': 16}, 'babelSynsetID': 'bn:00006882n', 'DBpediaURL': 'http://dbpedia.org/resource/Tellurium', 'BabelNetURL': 'http://babelnet.org/rdf/s00006882n', 'score': 0.0, 'coherenceScore': 0.0, 'globalScore': 0.0, 'source': 'MCS'}, {'tokenFragment': {'start': 1, 'end': 2}, 'charFragment': {'start': 15, 'end': 21}, 'babelSynsetID': 'bn:03332892n', 'DBpediaURL': 'http://dbpedia.org/resource/Tag_(game)', 'BabelNetURL': 'http://babelnet.org/rdf/s03332892n', 'score': 1.0, 'coherenceScore': 0.16666666666666666, 'globalScore': 0.010416666666666666, 'source': 'BABELFY'}, {'tokenFragment': {'start': 2, 'end': 2}, 'charFragment': {'start': 18, 'end': 21}, 'babelSynsetID': 'bn:00083181v', 'DBpediaURL': '', 'BabelNetURL': 'http://babelnet.org/rdf/s00083181v', 'score': 1.0, 'coherenceScore': 0.16666666666666666, 'globalScore': 0.010416666666666666, 'source': 'BABELFY'}, {'tokenFragment': {'start': 4, 'end': 4}, 'charFragment': {'start': 26, 'end': 29}, 'babelSynsetID': 'bn:00025364n', 'DBpediaURL': 'http://dbpedia.org/resource/Daughter', 'BabelNetURL': 'http://babelnet.org/rdf/s00025364n', 'score': 0.0, 'coherenceScore': 0.0, 'globalScore': 0.0, 'source': 'MCS'}, {
            'tokenFragment': {'start': 4, 'end': 8}, 'charFragment': {'start': 26, 'end': 45}, 'babelSynsetID': 'bn:00053325n', 'DBpediaURL': 'http://dbpedia.org/resource/Spanish_profanity', 'BabelNetURL': 'http://babelnet.org/rdf/s00053325n', 'score': 1.0, 'coherenceScore': 0.16666666666666666, 'globalScore': 0.010416666666666666, 'source': 'BABELFY'}, {'tokenFragment': {'start': 7, 'end': 7}, 'charFragment': {'start': 37, 'end': 40}, 'babelSynsetID': 'bn:00098342a', 'DBpediaURL': '', 'BabelNetURL': 'http://babelnet.org/rdf/s00098342a', 'score': 0.0, 'coherenceScore': 0.0, 'globalScore': 0.0, 'source': 'MCS'}, {'tokenFragment': {'start': 8, 'end': 8}, 'charFragment': {'start': 42, 'end': 45}, 'babelSynsetID': 'bn:00043009n', 'DBpediaURL': 'http://dbpedia.org/resource/Prostitution', 'BabelNetURL': 'http://babelnet.org/rdf/s00043009n', 'score': 1.0, 'coherenceScore': 0.5, 'globalScore': 0.15625, 'source': 'BABELFY'}]

        for result in sample:
            logger.debug(json.dumps(result, indent=2))
            tokenFragment = result.get('tokenFragment')
            tfStart = tokenFragment.get('start')
            tfEnd = tokenFragment.get('end')
            logger.debug(str(tfStart) + "\t" + str(tfEnd))

            # retrieving char fragment
            charFragment = result.get('charFragment')
            cfStart = charFragment.get('start')
            cfEnd = charFragment.get('end')
            logger.debug(str(cfStart) + "\t" + str(cfEnd))

            # retrieving BabelSynset ID
            synsetId = result.get('babelSynsetID')
            logger.debug(synsetId)

            self.call_synset_api(synsetId)


if __name__ == "__main__":
    # concept_embed_helper = ConceptnetEmbedding()
    # embedding_matrix = concept_embed_helper.get_embedding_matrix(
    #     create_embedding_dict=False)
    babelfy = BabelfyHelper()
    babelfy.read_sentences()
