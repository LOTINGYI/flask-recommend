import re
import requests
import pandas as pd
import nltk
nltk.download('stopwords')
import numpy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def open_df(sas_url):
    
    # response = requests.get(sas_url)
    # data = response.json()
    # df = pd.DataFrame.from_dict(data, orient='columns')
    df = pd.read_json(sas_url)
    df = df.drop(['identifier', 'can_scrape', 'url_saw','visit_count','status'], axis=1)

    for index, row in df.iterrows():
        if(re.findall(r'[\u4e00-\u9fff]+', row.descr)):
            df = df.drop(index)
        else:
            if(row.descr == 'Fail Connection' or row.descr == 'Connection Refused' or row.descr.isspace() == True):
                 df = df.drop(index)
    
    return df


def add(df, user,url,title,descr):
    df.loc[20000] = [user, url,title,descr]
    return df


def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def get_cosine_similarity(df):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(
                1, 3), min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(df['descr_clean'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities

def recommend(df, name, cosine_similarities):

    indices = pd.Series(df.index)
    recommended_results = []

    # gettin the index of the hotel that matches the name
    idx = indices[indices == name].index[0]
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending= False)
    # getting the indexes of the 10 most similar hotels except itself
    top_10_indexes = list(score_series.iloc[1:4].index)

    # populating the list with the names of the top 10 matching hotels
    for i in top_10_indexes:
        recommended_results.append(list(df.index)[i])

    return recommended_results
