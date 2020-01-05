import random
import pandas as pd
import numpy as np
import math
import ast
class KNN:
    def __init__(self, n_neighbours=1, distance='hamming'):
        self.n_neighbours = n_neighbours
        self.distance = distance

    def train(self, train_data):
        self.train_data = train_data

    def hamming_distance(self, test_data):
        predicted = list()
        train_data = self.train_data
        data_len = len(self.train_data)
        for i in range(0 , data_len):
            distance = 0
            train = train_data['vector'][i]
            for j in range(0, len(train)):
                if train[j] != test_data[j]:
                    distance += 1

            predicted.append([distance, train_data['class'][i]])
        predicted.sort(key = lambda x: x[0])

        prediction = predicted[0:self.n_neighbours]
        predicted_class = list()
        distance = list()
        for pred in prediction:
            distance.append(pred[0])
            predicted_class.append(pred[1])

        return predicted_class[np.argmax(predicted_class)]

    def euclidian_distance(self, test_data):
        predicted = list()
        train_data = self.train_data
        data_len = len(self.train_data)
        for i in range(0 , data_len):
            sum = 0
            train = ast.literal_eval(train_data['vector'][i])
            test = ast.literal_eval(test_data)

            for j in range(0, len(train)):
                if train[j] != test[j]:
                    #print(train[j], test[j])
                    sum += ((int(train[j])- int(test[j])) ** 2)

            predicted.append([math.sqrt(sum), train_data['class'][i]])
        predicted.sort(key = lambda x: x[0])

        prediction = predicted[0:self.n_neighbours]
        predicted_class = list()
        distance = list()
        for pred in prediction:
            distance.append(pred[0])
            predicted_class.append(pred[1])

        return predicted_class[np.argmax(predicted_class)]

    def cosine_distance(self, test_data):
        predicted = list()
        train_data = self.train_data
        data_len = len(self.train_data)
        for i in range(0 , data_len):
            similarity = 0
            train = ast.literal_eval(train_data['vector'][i])
            test = ast.literal_eval(test_data)

            dot_product = np.dot(train, test)

            test_squares = [ i**2 for i in test]
            train_squares = [i**2 for i in train]

            test_squares = np.sum(test_squares)
            train_squares = np.sum(train_squares)

            similarity = dot_product / (test_squares * train_squares)

            predicted.append([math.sqrt(similarity), train_data['class'][i]])
        predicted.sort(key = lambda x: x[0], reverse=True)

        prediction = predicted[0:self.n_neighbours]
        predicted_class = list()
        distance = list()
        for pred in prediction:
            distance.append(pred[0])
            predicted_class.append(pred[1])

        return predicted_class[np.argmax(predicted_class)]


    def predict(self, test_data):
        prediction = list()
        for i in range(0 , len(test_data)):
            if self.distance == 'hamming':
                prediction.append(self.hamming_distance(test_data['vector'][i]))
                print('Predicting ', i, 'Out of', len(test_data))
            elif self.distance == 'euclidian':
                prediction.append(self.euclidian_distance(test_data['vector'][i]))
                print('Predicting ', i, 'Out of', len(test_data))
            elif self.distance == 'cosine':
                prediction.append(self.cosine_distance(test_data['vector'][i]))
                print('Predicting ', i, 'Out of', len(test_data))
        return prediction
