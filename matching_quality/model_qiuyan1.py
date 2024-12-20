import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
from Autoencoder import *
from preprocessing import *
from datetime import datetime

import urllib3
urllib3.disable_warnings()

def preprocess(docs, samp_size=None):
    """
    Preprocess the data
    """
    if not samp_size:
        samp_size = 3

    print('Preprocessing raw texts ...')
    n_docs = len(docs)
    print(n_docs)
    sentences = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed
    idx_in = []  # index of sample selected
    #     samp = list(range(100))
    samp = n_docs
    for i in range(samp):
        sentence = preprocess_sent(docs[i])
        token_list = preprocess_word(sentence)
        if token_list:
            idx_in.append(i)
            sentences.append(sentence)
            token_lists.append(token_list)

        print('{} %'.format(str(np.round((i + 1) / samp * 100, 2))), end='\r')
    # print(token_lists)
    print('Preprocessing raw texts. Done!')
    return sentences, token_lists, idx_in



# define model object
class Topic_Model1:
    def __init__(self, k=10, method='TFIDF'):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        if method not in {'TFIDF', 'LDA', 'MPNet', 'LDA_MPNet'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = None
        self.corpus = None
        #         self.stopwords = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  # parameter for reletive importance of lda
        self.method = method
        self.AE = None
        self.id = method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vecotr representations from selected methods
        """
        # Default method
        if method is None:
            method = self.method

        # turn tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'TFIDF':
            print('Getting vector representations for TF-IDF ...')
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(sentences)
            print('Getting vector representations for TF-IDF. Done!')
            return vec

        elif method == 'LDA':
            print('Getting vector representations for LDA ...')
            if not self.ldamodel:
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20, random_state=0)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')
            return vec

        elif method == 'MPNet':

            print('Getting vector representations for MPNet ...')
            from sentence_transformers import SentenceTransformer
            # model = SentenceTransformer('bert-base-nli-max-tokens') ##已弃用
            # model = SentenceTransformer('paraphrase-albert-small-v2') ##基于ALBERT模型
            # model = SentenceTransformer('paraphrase-TinyBERT-L6-v2') ##基于TinyBERT模型
            model = SentenceTransformer('all-mpnet-base-v2') ##基于MPNet模型
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for MPNet. Done!')
            return vec

        elif method == 'LDA_MPNet':
        # else:
            vec_lda1 = self.vectorize(sentences, token_lists, method='LDA')
            lda = pd.DataFrame(vec_lda1)
            lda.to_csv('./vector/lda_vector.csv')

            vec_mpnet1 = self.vectorize(sentences, token_lists, method='MPNet')
            mpnet = pd.DataFrame(vec_mpnet1)
            mpnet.to_csv('./vector/MPNet_vector.csv')

            #LDA和MPNet向量连接
            vec_ldampnet1 = np.c_[vec_lda1, vec_mpnet1]
            ldampnet = pd.DataFrame(vec_ldampnet1)
            ldampnet.to_csv('./vector/ldampnet_vector.csv')
            return vec_ldampnet1

            # self.vec['LDA_BERT_FULL'] = vec_ldampnet1
            # if not self.AE:
            #     self.AE = Autoencoder()
            #     print('Fitting Autoencoder ...')
            #     self.AE.fit(vec_ldampnet1)
            #     print('Fitting Autoencoder Done!')
            #vec1 = self.AE.encoder.predict(vec_ldampnet1)
            # return vec1

    def fit(self, sentences, token_lists, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        # Default method
        if method is None:
            method = self.method
        # Default clustering method
        if m_clustering is None:
            m_clustering = KMeans

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        ####################################################
        #### Getting ldamodel or vector representations ####
        ####################################################

        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20, random_state=0)
                print('Fitting LDA Done!')
        else:
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            self.vec[method] = self.vectorize(sentences, token_lists, method)
            self.cluster_model.fit(self.vec[method])
            print('Clustering embeddings. Done!')

    # def predict(self, sentences, token_lists, out_of_sample=None):
    #     """
    #     Predict topics for new_documents
    #     """
    #     # Default as False
    #     out_of_sample = out_of_sample is not None
    #
    #     if out_of_sample:
    #         corpus = [self.dictionary.doc2bow(text) for text in token_lists]
    #         if self.method != 'LDA':
    #             vec = self.vectorize(sentences, token_lists)
    #             print(vec)
    #     else:
    #         corpus = self.corpus
    #         vec = self.vec.get(self.method, None)
    #
    #     if self.method == "LDA":
    #         lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
    #                                                  key=lambda x: x[1], reverse=True)[0][0],
    #                                 corpus)))
    #     else:
    #         lbs = self.cluster_model.predict(vec)
    #     return lbs
