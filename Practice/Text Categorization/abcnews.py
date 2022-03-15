from icecream import ic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

# Read data
data = pd.read_csv('../../data/abcnews-date-text.csv')
data.info()
desc = data['headline_text'].head(5000).values
ic(desc)

punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)
# ic(stop_words)
# ic(type(stop_words))

vectorizer = TfidfVectorizer(stop_words=stop_words)


def tokenizer(text):
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')
    new_text = [' '.join(tokenizer.tokenize(' '.join([stemmer.stem(w) for w in s.split(' ')]))) for s in text]
    return np.array(new_text)


ic(tokenizer(desc))
X = vectorizer.fit_transform(tokenizer(desc))
ic(vectorizer.get_feature_names_out())

# wcss = []
# for i in range(1, 12):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
#
# plt.plot(wcss)
# plt.xlabel('Cluster No.')
# plt.ylabel('WCSS')
# plt.title('Elbow')
# plt.savefig('Elbow.png')
# plt.show()

# 根据elbow人工选择cluster center数目
cluster_center = 5
kmeans = KMeans(n_clusters=cluster_center)
kmeans.fit(X)

output = pd.DataFrame(desc, columns=['headline_text'])
output['category'] = kmeans.labels_
# output.to_csv('abcnews.csv')
print(output)
