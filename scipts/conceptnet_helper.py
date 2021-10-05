import logging
import mmap
import os

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import pickle


logging.basicConfig(
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


if __name__ == "__main__":
    concept_embed_helper = ConceptnetEmbedding()
    embedding_matrix = concept_embed_helper.get_embedding_matrix(
        create_embedding_dict=False)
