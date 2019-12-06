import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import acquire

# normalizes string
def normalize(string):
    return unicodedata.normalize('NFKD',string)\
        .encode('ascii','ignore')\
        .decode('utf-8','ignore')

# removes special characters
def remove_special_characters(string):
    return re.sub(r"[^a-z0-9'\s]", '', string)

#sets all columns in df to lowercase, no special characters, and all ascii
def basic_clean(df):
    for c in df.columns:
        df[c] = df[c].astype(str).str.lower().apply(normalize)\
        .apply(remove_special_characters)
    return df

# tokenize string
def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string,return_str=True)


#stems all words in string and returns list of stemed words
def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = " ".join(stems)
    return stems, string

def lemmatize(string):
    snl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(stems)
    return lemmas, string

#removes stopwords from string)
def remove_stopwords(string, extra_words=[], exclude_words=[]):
    
    # Tokenize the string
    string = tokenize(string)

    words = string.split()
    stopword_list = stopwords.words('english')

    # remove the excluded words from the stopword list
    stopword_list = set(stopword_list) - set(exclude_words)

    # add in the user specified extra words
    stopword_list = stopword_list.union(set(extra_words))

    filtered_words = [w for w in words if w not in stopword_list]
    final_string = " ".join(filtered_words)
    return final_string