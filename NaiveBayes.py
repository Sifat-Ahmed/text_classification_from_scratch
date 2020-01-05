import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import math
from nltk import RegexpTokenizer, SnowballStemmer, PorterStemmer, word_tokenize
stopword_list = list(stopwords.words('english'))
stemmer = SnowballStemmer('english')
tokenizer = RegexpTokenizer(r'\w+')

class NaiveBayes:

    def __init__(self, smoothing_factor):
        self.wordmap = dict()
        self.bag = dict()
        self.vocabulary = set()
        self.priors = dict()
        self.class_proba = []
        self.smoothing_factor = smoothing_factor


    def calculate_prior(self,classes, train_data):
        for class_name in classes:
            self.priors[class_name] = len(train_data[train_data['class'] == class_name]) / len(train_data)
            self.wordmap[class_name] = {}
            self.bag[class_name] = list()

    def text_preprocess(self, text, training = True):
        words = tokenizer.tokenize(str(text))
        temp_words = list()
        for word in words:
            if word not in stopword_list and len(word) > 2:
                stemmed_word = stemmer.stem(word)
                if training:
                    self.vocabulary.add(word)
                temp_words.append(stemmed_word)
        return ' '.join(temp_words)

    def init_wordmap(self, class_name, text):
        for word in text.split():
           self.wordmap[class_name][word] = 0

    def wordcount_in_class(self, class_name, text):
        for word in text.split():
            self.wordmap[class_name][word] += 1

    def words_in_class(self, class_name):
        #print('Words in', class_name, len(self.bag[class_name]))
        return len(self.bag[class_name])

    def bag_of_words(self, class_name , text):
        for word in text.split():
            self.bag[class_name].append(word)

    def vocab_len(self):
        #print('Vocab len', len(self.vocabulary))
        return len(self.vocabulary)


    def train(self, train_data):
        self.classes = train_data['class'].unique()
        self.calculate_prior(self.classes, train_data)

        train_data['text'] = train_data['text'].map(lambda x: self.text_preprocess(x))
        #train_data.apply(lambda column: self.init_wordmap(column['class'], column['text']), axis = 1)
        #train_data.apply(lambda column: self.wordcount_in_class(column['class'], column['text']), axis = 1)
        #train_data.apply(lambda column: self.bag_of_words(column['class'], column['text']), axis = 1)

        for i in range(0, len(train_data)):
            self.init_wordmap(train_data['class'][i], train_data['text'][i])
            self.wordcount_in_class(train_data['class'][i], train_data['text'][i])
            self.bag_of_words(train_data['class'][i], train_data['text'][i])



    def predict_proba(self, text):
        class_proba = list()
        words = self.text_preprocess(text, training=False).split()
        for class_ in self.classes:
            prior = self.priors[class_]
            self.word_given_class = 1
            for word in words:
                if word in self.wordmap[class_].keys():
                    self.word_given_class *= math.log((self.wordmap[class_][word] + self.smoothing_factor) /\
                                   (self.words_in_class(class_) + self.vocab_len()))
            class_proba.append(self.word_given_class*prior)
        return class_proba

    def predict(self, test_data):
        prediction = list()
        for i in range(0 , len(test_data)):
            pred = self.predict_proba(test_data['text'][i])
            print('Predicting ', i, 'Out of', len(test_data))
            prediction.append(self.classes[np.argmax(pred)])

        return prediction
