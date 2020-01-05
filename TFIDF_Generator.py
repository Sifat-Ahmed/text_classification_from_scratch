import numpy as np
import pandas as pd
import math
from nltk.corpus import stopwords
from nltk import RegexpTokenizer, SnowballStemmer, PorterStemmer, word_tokenize

stopword_list = list(stopwords.words('english'))
stemmer = SnowballStemmer('english')
#stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

TRAIN_DATASET_PATH = 'Dataset\\training.csv'
TEST_DATASET_PATH = 'Dataset\\test.csv'
STOPWORDS_PATH = 'Dataset\\Stopwords.txt'

file = open(STOPWORDS_PATH, 'r', encoding='utf-8')

for line in file.readlines():
    word = line.strip()
    if word not in stopword_list:
        stopword_list.append(word)
file.close()

#print('Total stopwords', len(stopword_list))

train_dataset = pd.read_csv(TRAIN_DATASET_PATH , sep=',', index_col=False)
train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)

test_dataset = pd.read_csv(TEST_DATASET_PATH , sep=',', index_col=False)
test_dataset = test_dataset.sample(frac=1).reset_index(drop=True)



BAG_OF_WORDS = list()

def text_preprocess(text, training = True):
    words = tokenizer.tokenize(str(text))
    temp_words = list()
    for word in words:
        if word not in stopword_list and len(word) > 2:
            stemmed_word = stemmer.stem(word)
            if training:
                BAG_OF_WORDS.append(stemmed_word)
            temp_words.append(stemmed_word)
    return ' '.join(temp_words)

train_dataset['text'] = train_dataset['text'].map(lambda x: text_preprocess(x))
unique , counts = np.unique(BAG_OF_WORDS, return_counts=True)
Term_Frequency = dict(zip(unique, counts))
BAG_OF_WORDS = set(BAG_OF_WORDS)
Document_Frequency = dict()

for word in BAG_OF_WORDS:
    count = 0
    for i in range(0, len(train_dataset)):
        if word in train_dataset['text'][i].split():
            count += 1
    Document_Frequency[word] = count

TFIDF_vector = [0] * len(BAG_OF_WORDS)

WORD_MAP = dict()
index = 0
for word in BAG_OF_WORDS:
    WORD_MAP[word] = index
    index += 1


def get_vector(vector, word_map, text, total_documents, bow, TF_vector, IDF_vector):
    words = text.split(' ')
    for word in words:
        if word in word_map.keys():
            idf_score = math.log(total_documents/IDF_vector[word])
            tf_score = TF_vector[word] / len(bow)

            vector[word_map[word]] = tf_score * idf_score

    return vector

print(BAG_OF_WORDS)


train_dataset['vector'] = train_dataset['text'].map(lambda x: get_vector(TFIDF_vector.copy(), WORD_MAP, x, len(train_dataset), BAG_OF_WORDS, Term_Frequency, Document_Frequency))
train_dataset = train_dataset[['vector', 'class']]
train_dataset.to_csv('Vectors\TFIDF_train_vectors.csv', sep=',')

test_dataset['text'] = test_dataset['text'].map(lambda x: text_preprocess(x, training=False))
test_dataset['vector'] = test_dataset['text'].map(lambda x: get_vector(TFIDF_vector.copy(), WORD_MAP, x, len(train_dataset), BAG_OF_WORDS, Term_Frequency, Document_Frequency))
test_dataset = test_dataset[['vector', 'class']]
test_dataset.to_csv('Vectors\TFIDF_test_vectors.csv', sep=',')
