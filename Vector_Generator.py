import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk import RegexpTokenizer, SnowballStemmer, PorterStemmer, word_tokenize

stopword_list = list(stopwords.words('english'))
stemmer = SnowballStemmer('english')
#stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

TRAIN_DATASET_PATH = 'Dataset\\training.csv'
TEST_DATASET_PATH = 'Dataset\\test.csv'
STOPWORDS_PATH = 'Dataset\\Stopwords.txt'


train_dataset = pd.read_csv(TRAIN_DATASET_PATH , sep=',', index_col=False)
train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)

file = open(STOPWORDS_PATH, 'r', encoding='utf-8')

for line in file.readlines():
    word = line.strip()
    if word not in stopword_list:
        stopword_list.append(word)
file.close()

#print('Total stopwords', len(stopword_list))

def is_valid_word(word):
    # Check if word begins with an alphabet
    return re.search(r'^[a-zA-Z][a-zA-Z\._]*$', word) is not None


BAG_OF_WORDS = list()

def text_preprocess(text, training = True):
    words = tokenizer.tokenize(str(text))
    temp_words = list()
    for word in words:
        if is_valid_word(word):
            if word not in stopword_list and len(word) > 2:
                stemmed_word = stemmer.stem(word)
                if training:
                    BAG_OF_WORDS.append(stemmed_word)
                temp_words.append(stemmed_word)
    return ' '.join(temp_words)

train_dataset['text'] = train_dataset['text'].map(lambda x: text_preprocess(x))

BAG_OF_WORDS = set(BAG_OF_WORDS)

WORD_MAP = dict()
index = 0

for word in BAG_OF_WORDS:
    WORD_MAP[word] = index
    index += 1

print(WORD_MAP)

INITIAL_VECTOR = [0] * len(BAG_OF_WORDS)

def get_vector(vector, word_map, text):
    words = text.split(' ')
    for word in words:
        if word in word_map.keys():
            vector[word_map[word]] = 1
    return vector

print(BAG_OF_WORDS)

"""
train_dataset['vector'] = train_dataset['text'].map(lambda x: get_vector(INITIAL_VECTOR.copy(), WORD_MAP, x))
train_dataset = train_dataset[['vector', 'class']]
train_dataset.to_csv('hamming_train_vectors.csv', sep=',')

test_dataset = pd.read_csv(TEST_DATASET_PATH , sep=',', index_col=False)
test_dataset = test_dataset.sample(frac=1).reset_index(drop=True)
test_dataset['text'] = test_dataset['text'].map(lambda x: text_preprocess(x, False))

test_dataset['vector'] = test_dataset['text'].map(lambda x: get_vector(INITIAL_VECTOR.copy(), WORD_MAP, x))
test_dataset = test_dataset[['vector', 'class']]
test_dataset.to_csv('hamming_test_vectors.csv', sep=',')

word_file = open('word_map', 'wb')
bag_file = open('word_bag', 'wb')

pickle.dump(WORD_MAP, word_file)
pickle.dump(BAG_OF_WORDS, bag_file)

word_file.close()
bag_file.close()
"""
