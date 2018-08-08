import pandas as pd
import numpy as np
import textblob
from textblob.classifiers import NaiveBayesClassifier

def read_data(filepathtr, filepathts):
    train = pd.read_csv(filepathtr)
    test = pd.read_csv(filepathts)
    train.dropna(subset=['type'], inplace=True)
    train['text_type'] = tuple(zip(train['text'].astype(str), train['type'].astype(str)))
    train = train['text_type'].tolist()
    test['type'] = np.nan
    test['text_type'] = tuple(zip(test['text'].astype(str), test['type'].astype(str)))
    test = test['text_type'].tolist()
    return train, test

def classify(train):
    cl = NaiveBayesClassifier(train)
    return cl

def accuracy(cl, test):
    acc = cl.accuracy(test)
    return acc
    print(acc)

if __name__ == "__main__":
    train, test = read_data('train_small_.csv', 'test.csv')
    format_data(train,test)
    classify(train)
    accuracy(cl,test)
