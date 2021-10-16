import csv
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import pandas as pd
#import pathlib
import sklearn.datasets
import matplotlib.pyplot as plt
from collections import Counter

def load_dataset(file):
    try: 
        data_dir = sklearn.datasets.load_files(file, shuffle=False, encoding='latin1')
        #print(data_dir)
        print("\n===> Classes: " + str(data_dir.target_names))
        #print(data_dir.target)
    except:
        print("File not found.")
        sys.exit(0)
    return data_dir


def plot_distribution(dataset_target, min_ticks, max_ticks):
    # Count the frequency of files in each classes
    freq_data = Counter(target for target in dataset_target)
    print("\n===> " + str(freq_data))
    business, entertainment, politics, sport, tech = freq_data[0], freq_data[1], freq_data[2], freq_data[3], freq_data[4]
    classes = [business, entertainment, politics, sport, tech]
    labels = ('business', 'entertainment', 'politics', 'sport', 'tech')
    plt.xticks(range(len(classes)), labels)
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.ylim(min_ticks, max_ticks)
    plt.title('Number of files in each classes')
    plt.bar(range(len(classes)), classes)
    # Annotating the bar plot with the values (total label count)
    for i in range(len(labels)):
        plt.annotate(classes[i], (-0.06 + i, classes[i] + 2))

    plt.show()
    plt.savefig('BBC-distribution.png')


# Vectorize dataset
def preprocess_dataset(dataset_data):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dataset_data)
    count_array = count_matrix.toarray()
    print("\n===> Ouput feature names: " + str(vectorizer.get_feature_names_out()))
    print("\n===> Array of word frequency:\n" + str(count_array))
    
    # Visualize matrix
    print("\n\n===> Visualize matrix of word frequency:\n")
    df = pd.DataFrame(data = count_array,columns = vectorizer.get_feature_names_out())
    print(df)
    return count_matrix


def split_dataset(dataset_data, dataset_target):
    x_train, x_test, y_train, y_test = train_test_split(dataset_data, dataset_target, test_size=0.2, random_state=None) # no shuffling before dataset split
    print("\n\n===> Dataset splitted")
    print("     x_train size: " + str(x_train.shape))
    print("     y_train size: " + str(y_train.shape))
    print("     x_test size: " + str(x_test.shape))
    print("     y_test size: " + str(y_test.shape))
    return x_train, x_test, y_train, y_test


def train_nb(x_train, y_train):
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train) # train it!
    return classifier

def test_nb(x_test, y_test, classifier):
    pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
    print(accuracy)


data_dir = load_dataset('BBC')
plot_distribution(data_dir.target, 350, 550)
processed_data = preprocess_dataset(data_dir.data)
x_train, x_test, y_train, y_test = split_dataset(processed_data, data_dir.target)
classifier = train_nb(x_train, y_train)
test_nb(x_test, y_test, classifier)
