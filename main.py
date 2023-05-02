import pickle
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy import sparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from sklearn.cluster import KMeans


class Transformation:

    def __init__(self, data):
        self.data = data
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False
        )

        self.lda = LatentDirichletAllocation(
            n_components=10,
            learning_method='online',
            learning_offset=50,
            random_state=0
        )
        self.tfidf_matrix = None
        self.lda_matrix = None
        self.abstracts = [paper['abstract'] for paper in self.data.values() if paper['abstract'] is not None]


    def Tfidf(self):
        print('tfidf ran')
        x = self.tfidf.fit(self.abstracts)
        self.tfidf_matrix = self.tfidf.transform(self.abstracts)
        with open(r'C:\Users\imran\PycharmProjects\KnowledgeMeter\models\topic_modeling\tfidf.pkl', 'wb') as f:
            pickle.dump(self.tfidf, f)
        return self.tfidf_matrix

    def Lda(self, n_components=10):
        print('lda ran')
        self.Tfidf()
        self.lda_matrix = self.lda.fit_transform(self.tfidf_matrix)
        with open(r'C:\Users\imran\PycharmProjects\KnowledgeMeter\models\topic_modeling\lda.pkl', 'wb') as f:
            pickle.dump(self.lda, f)

        return self.lda_matrix