import csv
import pandas as pd
import numpy as np
from KNN import KNN
from NaiveBayes import NaiveBayes
from sklearn.metrics import accuracy_score
import ast
"""
file = open('KNN.txt', 'a')

neighbours = 5
distance = 'cosine'
knn_classifier = KNN(n_neighbours = neighbours, distance = distance)

if knn_classifier.distance == 'hamming':
    TRAIN_DATASET = 'Vectors\\hamming_train_vectors.csv'
    TEST_DATASET = 'Vectors\\hamming_test_vectors.csv'

elif knn_classifier.distance == 'euclidian':
    TRAIN_DATASET = 'Vectors\\euclidian_train_vectors.csv'
    TEST_DATASET = 'Vectors\\euclidian_test_vectors.csv'

else:
    TRAIN_DATASET = 'Vectors\\TFIDF_train_vectors.csv'
    TEST_DATASET = 'Vectors\\TFIDF_test_vectors.csv'


train_data = pd.read_csv(TRAIN_DATASET, sep = ',')
test_data = pd.read_csv(TEST_DATASET, sep = ',')

knn_classifier.train(train_data)
prediction = knn_classifier.predict(test_data[0:50])
print(len(prediction), len(test_data[0:50]))
accuracy = accuracy_score(prediction, test_data['class'][0:50])
file.write('KNN - '+ knn_classifier.distance +' - '+ str(neighbours)+' Accuracy ' + str(accuracy)+' on '+ str(len(test_data[0:50])) + ' Samples. ' +'\n')

file.close()

"""
file = open('NB1.txt', 'a')
TRAIN_DATASET = 'Dataset\\training.csv'
TEST_DATASET = 'Dataset\\test.csv'

train_data = pd.read_csv(TRAIN_DATASET, sep = ',')
test_data = pd.read_csv(TEST_DATASET, sep = ',')

smoothing = 0.0

for i in range(0, 100):

    NB = NaiveBayes(smoothing_factor=smoothing)
    NB.train(train_data)
    prediction = NB.predict(test_data[:400])
    #print('Accuracy', accuracy_score(prediction, test_data['class']))
    accuracy = accuracy_score(prediction, test_data['class'][:400])

    file.write('NB - '+ str(NB.smoothing_factor) +' - '+' Accuracy ' + str(accuracy)+' on '+ str(len(test_data[:400])) + ' Samples. ' +'\n')
    smoothing += 0.02

file.close()
#"""
