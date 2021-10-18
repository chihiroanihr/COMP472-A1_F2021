import csv
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def load_dataset(file):
    try:
        data_dir = sklearn.datasets.load_files(file, shuffle=False, encoding='latin1')
        # print(data_dir)
        print("\n===> Classes: " + str(data_dir.target_names))
        # print(data_dir.target)
    except:
        print("File not found.")
        sys.exit(0)
    return data_dir


def plot_distribution(dataset_target, min_ticks, max_ticks):
    # Count the frequency of files in each classes
    freq_data = Counter(target for target in dataset_target)
    print("===> " + str(freq_data))
    business, entertainment, politics, sport, tech = freq_data[0], freq_data[1], freq_data[2], freq_data[3], freq_data[4]
    classes = [business, entertainment, politics, sport, tech]
    labels = ('business', 'entertainment', 'politics', 'sport', 'tech')
    # Plot
    plt.xticks(range(len(classes)), labels)
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.ylim(min_ticks, max_ticks)
    plt.title('Number of files in each classes')
    plt.bar(range(len(classes)), classes)
    # Annotating the bar plot with the values (total label count)
    for i in range(len(labels)):
        plt.annotate(classes[i], (-0.06 + i, classes[i] + 2))
    # Save graph
    plt.savefig('BBC-distribution.pdf')
    # Show graph
    plt.show()


# Vectorize dataset (convert text documents into numerical features)
def preprocess_dataset(dataset_data):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dataset_data)  # Fit the bag-of-words model
    # Get unique words/tokens found in all the documents (These represents the features)
    print("\n===> Ouput feature names: " + str(vectorizer.get_feature_names_out()))
    count_array = count_matrix.toarray()  # Numerical feature vector
    print("===> Array of word frequency:\n" + str(count_array))
    # Visualize matrix (numerical feature vector)
    df = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())
    print("===> Visualize matrix of word frequency:")
    print(df)
    return count_matrix, vectorizer


def split_dataset(dataset_data, dataset_target):
    x_train, x_test, y_train, y_test = train_test_split(dataset_data, dataset_target, test_size=0.2, random_state=None)  # no shuffling before dataset split
    print("\n\n===> Dataset splitted")
    print("     x_train size: " + str(x_train.shape))
    print("     y_train size: " + str(y_train.shape))
    print("     x_test size: " + str(x_test.shape))
    print("     y_test size: " + str(y_test.shape))
    return x_train, x_test, y_train, y_test


def train_nb(x_train, y_train, smoothing=1.0): # Default smoothing value: 1
    classifier = MultinomialNB(alpha=smoothing)
    classifier.fit(x_train, y_train)  # train it!
    return classifier


def test_nb(x_test, y_test, classifier):
    x_test_pred = classifier.predict(x_test)
    conf_matrix = confusion_matrix(y_test, x_test_pred)
    report = classification_report(y_test, x_test_pred)
    acc_score = round(accuracy_score(y_test, x_test_pred), 4)
    f1_macro_score = round(f1_score(y_test, x_test_pred, average='macro'), 4)
    f1_weighted_score = round(f1_score(y_test, x_test_pred, average='weighted'), 4)
    print("\n\n===> Result Prediction (Actual value & Predicted value):\n")
    print("---> Confusion matrix: \n" + str(conf_matrix) + "\n")
    print("---> Classification report (Precision, Recall, F1): \n\n" + str(report))
    print("---> Accuracy score: " + str(acc_score) + " (" + str(round(acc_score*100, 2)) + "%)")
    print("---> F1-score with macro average: " + str(f1_macro_score) + " (" + str(round(f1_macro_score*100, 2)) + "%)")
    print("---> F1-score with weighted average: " + str(f1_weighted_score) + " (" + str(round(f1_weighted_score*100, 2)) + "%)")
    return conf_matrix, report, acc_score, f1_macro_score, f1_weighted_score


# calculate prior probability on training set
def calculate_prior_prob(dataset_target):
    # Divde the total number of articles in one category by the total number of articles in all categories
    freq_data = Counter(target for target in dataset_target)
    all_counter = freq_data[0] + freq_data[1] + freq_data[2] + freq_data[3] + freq_data[4]
    counter0, counter1, counter2, counter3, counter4 = freq_data[0], freq_data[1], freq_data[2], freq_data[3], freq_data[4]
    prior0 = round(counter1/all_counter, 4)
    prior1 = round(counter1/all_counter, 4)
    prior2 = round(counter2/all_counter, 4)
    prior3 = round(counter3/all_counter, 4)
    prior4 = round(counter4/all_counter, 4)
    print("\n\n===> Prior Probabilities: ")
    print("     for business (0): " + str(prior0))
    print("     for entertainment (1): " + str(prior1))
    print("     for politics (2): " + str(prior2))
    print("     for sport (3): " + str(prior3))
    print("     for tech (4): " + str(prior4))
    return prior0, prior1, prior2, prior3, prior4


# Get unique words/tokens found in all the documents (These represents the features)
def size_vocabulary(dataset_data):
    # Num of columns = num unique features (tokens)
    size_all_vocabulary = dataset_data.shape[1]
    # Get unique words/tokens found in all the documents (These represents the features)
    print("\n\n===> Size of vocabulary (number of different words): " + str(size_all_vocabulary))
    return size_all_vocabulary


def num_word_tokens_each_class(trainset_data_clf):
    # classifier.feature_count_ => returns ndarray of shape (n_classes, n_features)
    # In other words, returns frequency of each features in each classes
    # ex)    [[  0. 192.   0. ...   0.   0.   0.]
    #        [  1. 112.   0. ...   1.   0.   2.]
    #        [  1. 144.   1. ...   0.   0.   0.]
    #        [  2.  19.   0. ...   0.   4.   0.]
    #        [  1. 154.   0. ...   0.   0.   0.]]
    row_sums = trainset_data_clf.sum(axis=1)  # Compute sum of row (each class)
    num_tokens0 = row_sums[0]
    num_tokens1 = row_sums[1]
    num_tokens2 = row_sums[2]
    num_tokens3 = row_sums[3]
    num_tokens4 = row_sums[4]
    return num_tokens0, num_tokens1, num_tokens2, num_tokens3, num_tokens4


def output_num_word_tokens_each_class(num_tokens0, num_tokens1, num_tokens2, num_tokens3, num_tokens4):
    print("\n===> Number of word-tokens in each classes: ")
    print("     for business (0): " + str(num_tokens0))
    print("     for entertainment (1): " + str(num_tokens1))
    print("     for politics (2): " + str(num_tokens2))
    print("     for sport (3): " + str(num_tokens3))
    print("     for tech (4): " + str(num_tokens4))


def num_word_tokens_entire_corpus(trainset_data, testset_data):
    entire_num_word_tokens = trainset_data.toarray().sum() + testset_data.toarray().sum()
    print("\n===> Number of word-tokens in entire corpus: " + str(entire_num_word_tokens))
    return entire_num_word_tokens


def num_zero_freq_words_each_classes(trainset_data_clf):
    # compute number of words with 0 frequency from each classes
    row_zero_freq_words = np.count_nonzero(trainset_data_clf == 0, axis=1)  # arr==0 -> Count 0
    num_zero0 = row_zero_freq_words[0]
    num_zero1 = row_zero_freq_words[1]
    num_zero2 = row_zero_freq_words[2]
    num_zero3 = row_zero_freq_words[3]
    num_zero4 = row_zero_freq_words[4]
    # compute entire number of words from each classes
    num_tokens0, num_tokens1, num_tokens2, num_tokens3, num_tokens4 = num_word_tokens_each_class(trainset_data_clf)
    # Percentage of words with 0 frequency = number of words with 0 frequency / entire number of words in each classes
    percent_zero0 = round(num_zero0/num_tokens0 * 100, 2)
    percent_zero1 = round(num_zero1/num_tokens1 * 100, 2)
    percent_zero2 = round(num_zero2/num_tokens2 * 100, 2)
    percent_zero3 = round(num_zero3/num_tokens3 * 100, 2)
    percent_zero4 = round(num_zero4/num_tokens4 * 100, 2)
    print("\n===> Number of word-tokens with zero frequency in each classes: ")
    print("     for business (0): number = " + str(num_zero0) + ", percentage = " + str(percent_zero0) + "%")
    print("     for entertainment (1): number = " + str(num_zero1) + ", percentage = " + str(percent_zero1) + "%")
    print("     for politics (2): number = " + str(num_zero2) + ", percentage = " + str(percent_zero2) + "%")
    print("     for sport (3): number = " + str(num_zero3) + ", percentage = " + str(percent_zero3) + "%")
    print("     for tech (4): number = " + str(num_zero4) + ", percentage = " + str(percent_zero4) + "%")
    return num_zero0, num_zero1, num_zero2, num_zero3, num_zero4, percent_zero0, percent_zero1, percent_zero2, percent_zero3, percent_zero4


def num_one_freq_words_entire_corpus(trainset_data, testset_data, entire_num_word_tokens):
    trainset_data = trainset_data.toarray()
    testset_data = testset_data.toarray()
    num_entire_one_freq_words = np.count_nonzero(trainset_data == 1) + np.count_nonzero(testset_data == 1)
    percent_entire_one_freq_words = round((num_entire_one_freq_words / entire_num_word_tokens) * 100, 2)
    print("\n===> Number of word-tokens with one frequency in entire corpus = " +
          str(num_entire_one_freq_words) + ", percentage = " + str(percent_entire_one_freq_words) + "%")
    return num_entire_one_freq_words, percent_entire_one_freq_words


def log_prob_of_fav_words(fav_word1, fav_word2, vectorizer, trainset_data_clf, size_all_vocabulary):
    # convert ndarray into list for indexing
    list_feature_names = vectorizer.get_feature_names_out().tolist()
    # find indices of 2 favorite words from list of feature names (from 29421 unique features)
    index_fav_word1 = list_feature_names.index(fav_word1)
    index_fav_word2 = list_feature_names.index(fav_word2)
    print("\n\n===> Index of fav word1 \"" + fav_word1 + "\": " + str(index_fav_word1) +
          ", and Index of fav word2 \"" + fav_word2 + "\": " + str(index_fav_word2))
    # make list of favorite word frequencies in each classes which can be found by looping vector of features
    list_freq_fav_word1 = []
    list_freq_fav_word2 = []
    for row in trainset_data_clf:
        list_freq_fav_word1.append(row[index_fav_word1])
        list_freq_fav_word2.append(row[index_fav_word2])
    print("===> Frequency of fav word1 \"" + fav_word1 + "\" in all classes: " + str(list_freq_fav_word1) +
          ", and Frequency of fav word2 \"" + fav_word2 + "\" in all classes: " + str(list_freq_fav_word2))
    # compute entire number of words from each classes
    num_tokens0, num_tokens1, num_tokens2, num_tokens3, num_tokens4 = num_word_tokens_each_class(trainset_data_clf)
    alpha = 1  # default value (smoothing value)
    # Log prob = (freq_fav_word + alpha) / (all_words_in_class + alpha * unique_vocabulary_size)
    log_prob_fav1_given_class0 = (list_freq_fav_word1[0] + alpha) / (num_tokens0 + alpha * size_all_vocabulary)
    log_prob_fav1_given_class1 = (list_freq_fav_word1[1] + alpha) / (num_tokens1 + alpha * size_all_vocabulary)
    log_prob_fav1_given_class2 = (list_freq_fav_word1[2] + alpha) / (num_tokens2 + alpha * size_all_vocabulary)
    log_prob_fav1_given_class3 = (list_freq_fav_word1[3] + alpha) / (num_tokens3 + alpha * size_all_vocabulary)
    log_prob_fav1_given_class4 = (list_freq_fav_word1[4] + alpha) / (num_tokens4 + alpha * size_all_vocabulary)
    log_prob_fav2_given_class0 = (list_freq_fav_word2[0] + alpha) / (num_tokens0 + alpha * size_all_vocabulary)
    log_prob_fav2_given_class1 = (list_freq_fav_word2[1] + alpha) / (num_tokens1 + alpha * size_all_vocabulary)
    log_prob_fav2_given_class2 = (list_freq_fav_word2[2] + alpha) / (num_tokens2 + alpha * size_all_vocabulary)
    log_prob_fav2_given_class3 = (list_freq_fav_word2[3] + alpha) / (num_tokens3 + alpha * size_all_vocabulary)
    log_prob_fav2_given_class4 = (list_freq_fav_word2[4] + alpha) / (num_tokens4 + alpha * size_all_vocabulary)
    print("===> Log probability of word1 (\"" + fav_word1 + "\") with smoothing value = 1: ")
    print("     log(P(\"japan\"|business)) = " + str(log_prob_fav1_given_class0))
    print("     log(P(\"japan\"|entertainment)) = " + str(log_prob_fav1_given_class1))
    print("     log(P(\"japan\"|politics)) = " + str(log_prob_fav1_given_class2))
    print("     log(P(\"japan\"|sport)) = " + str(log_prob_fav1_given_class3))
    print("     log(P(\"japan\"|tech)) = " + str(log_prob_fav1_given_class4))
    print("===> Log probability of word2 (\"" + str(fav_word2 + "\") with smoothing value = 1: "))
    print("     log(P(\"korea\"|business)) = " + str(log_prob_fav2_given_class0))
    print("     log(P(\"korea\"|entertainment)) = " + str(log_prob_fav2_given_class1))
    print("     log(P(\"korea\"|politics)) = " + str(log_prob_fav2_given_class2))
    print("     log(P(\"korea\"|sport)) = " + str(log_prob_fav2_given_class3))
    print("     log(P(\"korea\"|tech)) = " + str(log_prob_fav2_given_class4))
    return log_prob_fav1_given_class0, log_prob_fav1_given_class1, log_prob_fav1_given_class2, log_prob_fav1_given_class3, log_prob_fav1_given_class4, log_prob_fav2_given_class0, log_prob_fav2_given_class1, log_prob_fav2_given_class2, log_prob_fav2_given_class3, log_prob_fav2_given_class4


################ MAIN ################

# Reset file content for each execution
file = open("bbc-performance.txt","w")
file.close()

# 3. Load the BBC dataset (Classes:  business,entertainment, politics, sport, tech)
data_dir = load_dataset('BBC')
# 2. Plot the distribution of the instances in each class and save the graphic in a file "BBC-distribution.pdf"
plot_distribution(data_dir.target, 350, 550)
# 4. Pre-process the dataset to have the features ready to be used by a multinomial Naive Bayes classifier
# This means that the frequency of each word in each class must be computed and stored in a term-document matrix
processed_data, vectorizer = preprocess_dataset(data_dir.data)
# 5. Split the dataset into 80% for training and 20% for testing
x_train, x_test, y_train, y_test = split_dataset(processed_data, data_dir.target)

for i in range(4):
    # 6. Train a multinomial Naive Bayes Classifier on the training set using the default parameters
    if i == 2:
        print("\n\n\n---------- Multi-nomial NB smoothing value = 0.0001 ----------")
        classifier = train_nb(x_train, y_train, 0.0001)
    elif i == 3:
        print("\n\n\n---------- Multi-nomial NB smoothing value = 0.9 ----------")
        classifier = train_nb(x_train, y_train, 0.9)
    else:
        print("\n\n\n---------- Multi-nomial NB default values, try " + str((i + 1)) + " ----------")
        classifier = train_nb(x_train, y_train)
    # And evaluate it on the test set:
    # (b)  the confusion matrix
    # (c)  the precision, recall, and F1-measure for each class
    # (d)  the accuracy, macro-average F1 and weighted-average F1 of the model
    conf_matrix, report, acc_score, f1_macro_score, f1_weighted_score = test_nb(x_test, y_test, classifier)
    # (e)  the prior probability of each class IN TRAIN SET
    prior0, prior1, prior2, prior3, prior4 = calculate_prior_prob(y_train)
    # (f)  the size of the vocabulary IN TRAIN SET (i.e.  the number of different words)
    size_all_vocabulary = size_vocabulary(x_train)
    # (g)  the number of word-tokens in each class IN TRAIN SET (i.e.  the number of words in total)
    num_tokens0, num_tokens1, num_tokens2, num_tokens3, num_tokens4 = num_word_tokens_each_class(classifier.feature_count_.astype(int))
    output_num_word_tokens_each_class(num_tokens0, num_tokens1, num_tokens2, num_tokens3, num_tokens4)
    # (h)  the number of word-tokens in the entire corpus (TRAIN SET + TEST SET)
    entire_num_word_tokens = num_word_tokens_entire_corpus(x_train, x_test)
    # (i)  the number and percentage of words with a frequency of zero in each class IN TRAIN SET
    num_zero0, num_zero1, num_zero2, num_zero3, num_zero4, percent_zero0, percent_zero1, percent_zero2, percent_zero3, percent_zero4 = num_zero_freq_words_each_classes(classifier.feature_count_.astype(int))
    # (j)  the number and percentage of words with a frequency of one in the entire corpus (TRAIN SET + TEST SET)
    num_entire_one_freq_words, percent_entire_one_freq_words = num_one_freq_words_entire_corpus(x_train, x_test, entire_num_word_tokens)
    # (k)  your 2 favorite words (that are present in the vocabulary) and their log-prob IN TRAIN SET
    fav_word1, fav_word2 = "japan", "korea"
    log_prob_fav1_given_class0, log_prob_fav1_given_class1, log_prob_fav1_given_class2, log_prob_fav1_given_class3, log_prob_fav1_given_class4, log_prob_fav2_given_class0, log_prob_fav2_given_class1, log_prob_fav2_given_class2, log_prob_fav2_given_class3, log_prob_fav2_given_class4 = log_prob_of_fav_words(fav_word1, fav_word2, vectorizer, classifier.feature_count_.astype(int), size_all_vocabulary)


    ### Output files for result ###
    with open('bbc-performance.txt', 'a') as file:
        if i == 2:
            file.write("---------- Multi-nomial NB smoothing value = 0.0001 ----------\n\n")
        elif i == 3:
            file.write("---------- Multi-nomial NB smoothing value = 0.9 ----------\n\n")
        else:
            file.write("---------- Multi-nomial NB default values, try " + str((i+1)) + " ----------\n\n")

        file.write("Result Prediction (Actual value & Predicted value)\n")
        file.write("(b) Confusion matrix: \n" + str(conf_matrix) + "\n")
        file.write("(c) Classification report (Precision, Recall, F1): \n" + str(report) + "\n")
        file.write("(d) Accuracy score: " + str(acc_score) + " (" + str(round(acc_score*100, 2)) + "%)\n\n")
        file.write("    F1-score with macro average: " + str(f1_macro_score) + " (" + str(round(f1_macro_score*100, 2)) + "%)\n")
        file.write("    F1-score with weighted average: " + str(f1_weighted_score) + " (" + str(round(f1_weighted_score*100, 2)) + "%)\n")

        file.write("(e) The Prior Probability of each class IN TRAIN SET\n")
        file.write("    for business (0): " + str(prior0) + "\n")
        file.write("    for entertainment (1): " + str(prior1) + "\n")
        file.write("    for politics (2): " + str(prior2) + "\n")
        file.write("    for sport (3): " + str(prior3) + "\n")
        file.write("    for tech (4): " + str(prior4) + "\n")

        file.write("(f) The size of the vocabulary IN TRAIN SET (i.e. the number of different words): " + str(size_all_vocabulary) + "\n")

        file.write("(g) The number of word-tokens in each class IN TRAIN SET (i.e.  the number of words in total)\n")
        file.write("    for business (0): " + str(num_tokens0) + "\n")
        file.write("    for entertainment (1): " + str(num_tokens1) + "\n")
        file.write("    for politics (2): " + str(num_tokens2) + "\n")
        file.write("    for sport (3): " + str(num_tokens3) + "\n")
        file.write("    for tech (4): " + str(num_tokens4) + "\n")

        file.write("(h) The number of word-tokens in the entire corpus (TRAIN SET + TEST SET): " + str(entire_num_word_tokens) + "\n")

        file.write("(i) The number and percentage of 'words with a frequency of zero' in each class IN TRAIN SET\n")
        file.write("    for business (0): number = " + str(num_zero0) + ", percentage = " + str(percent_zero0) + "%\n")
        file.write("    for entertainment (1): number = " + str(num_zero1) + ", percentage = " + str(percent_zero1) + "%\n")
        file.write("    for politics (2): number = " + str(num_zero2) + ", percentage = " + str(percent_zero2) + "%\n")
        file.write("    for sport (3): number = " + str(num_zero3) + ", percentage = " + str(percent_zero3) + "%\n")
        file.write("    for tech (4): number = " + str(num_zero4) + ", percentage = " + str(percent_zero4) + "%\n")

        file.write("(j) The number and percentage of 'words with a frequency of one' in the entire corpus (TRAIN SET + TEST SET):\n")
        file.write("    number = " + str(num_entire_one_freq_words) + ", percentage = " + str(percent_entire_one_freq_words) + "%\n")

        file.write(
            "(k) Your 2 favorite words (that are present in the vocabulary) and their log-prob IN TRAIN SET\n")
        file.write(" 1. Log probability of word1 (\"" + fav_word1 + "\") with smoothing value = 1: \n")
        file.write("    log(P(\"japan\"|business)) = " + str(log_prob_fav1_given_class0) + "\n")
        file.write("    log(P(\"japan\"|entertainment)) = " + str(log_prob_fav1_given_class1) + "\n")
        file.write("    log(P(\"japan\"|politics)) = " + str(log_prob_fav1_given_class2) + "\n")
        file.write("    log(P(\"japan\"|sport)) = " + str(log_prob_fav1_given_class3) + "\n")
        file.write("    log(P(\"japan\"|tech)) = " + str(log_prob_fav1_given_class4) + "\n")
        file.write(" 2. Log probability of word2 (\"" + str(fav_word2 + "\") with smoothing value = 1: \n"))
        file.write("    log(P(\"korea\"|business)) = " + str(log_prob_fav2_given_class0) + "\n")
        file.write("    log(P(\"korea\"|entertainment)) = " + str(log_prob_fav2_given_class1) + "\n")
        file.write("    log(P(\"korea\"|politics)) = " + str(log_prob_fav2_given_class2) + "\n")
        file.write("    log(P(\"korea\"|sport)) = " + str(log_prob_fav2_given_class3) + "\n")
        file.write("    log(P(\"korea\"|tech)) = " + str(log_prob_fav2_given_class4) + "\n")

        file.write("\n\n\n")