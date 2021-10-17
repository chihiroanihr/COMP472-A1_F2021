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
    # Show graph
    plt.show()
    # Save graph
    plt.savefig('BBC-distribution.png')


def preprocess_dataset(dataset_data): # Vectorize dataset (convert text documents into numerical features)
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(dataset_data) # Fit the bag-of-words model
    print("\n===> Ouput feature names: " + str(vectorizer.get_feature_names_out())) # Get unique words/tokens found in all the documents (These represents the features)
    count_array = count_matrix.toarray() # Numerical feature vector
    print("===> Array of word frequency:\n" + str(count_array))
    df = pd.DataFrame(data = count_array, columns = vectorizer.get_feature_names_out()) # Visualize matrix (numerical feature vector)
    print("===> Visualize matrix of word frequency:")
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
    '''
    print(classifier.class_log_prior_)
    print(classifier.class_count_)
    print(classifier.classes_)
    print(classifier.feature_count_)
    print(classifier.feature_log_prob_)
    print(classifier.n_features_in_)
    '''
    return classifier

def test_nb(x_test, y_test, classifier):
    x_test_pred = classifier.predict(x_test)
    conf_matrix = confusion_matrix(y_test, x_test_pred)
    report = classification_report(y_test, x_test_pred)
    acc_score = accuracy_score(y_test, x_test_pred)
    f1_macro_score = f1_score(y_test, x_test_pred, average='macro')
    f1_weighted_score = f1_score(y_test, x_test_pred, average='weighted')
    print("\n\n===> Result Prediction (Actual value & Predicted value):\n")
    print("---> Confusion matrix: \n" + str(conf_matrix) + "\n")
    print("---> Classification report (Precision, Recall, F1): \n\n" + str(report))
    print("---> Accuracy score: " + str(acc_score))
    print("---> F1-score with macro average: " + str(f1_macro_score))
    print("---> F1-score with weighted average: " + str(f1_weighted_score))
    return conf_matrix, report, acc_score, f1_macro_score, f1_weighted_score


def calculate_prior_prob(dataset_target): # calculate prior probability on training set
    # Doivde the total number of articles in one category by the total number of articles in all categories
    freq_data = Counter(target for target in dataset_target)
    all_counter = freq_data[0] + freq_data[1] + freq_data[2] + freq_data[3] + freq_data[4]
    counter0, counter1, counter2, counter3, counter4 = freq_data[0], freq_data[1], freq_data[2], freq_data[3], freq_data[4]
    prior0 = counter1/all_counter
    prior1 = counter1/all_counter
    prior2 = counter2/all_counter
    prior3 = counter3/all_counter
    prior4 = counter4/all_counter
    print("\n\n===> Prior Probabilities: ")
    print("---> for business (0): " + str(prior0))
    print("---> for entertainment (1): " + str(prior1))
    print("---> for politics (2): " + str(prior2))
    print("---> for sport (3): " + str(prior3))
    print("---> for tech (4): " + str(prior4))
    return prior0, prior1, prior2, prior3, prior4

def size_vocabulary(dataset_data):
    size_vocabulary = dataset_data.shape[1] # Num of columns = num unique features (tokens)
    print("\n===> Size of vocabulary (number of different words): " + str(size_vocabulary))# Get unique words/tokens found in all the documents (These represents the features)

def num_word_tokens(dataset_data):
    entire_num_word_tokens = dataset_data.toarray().sum()
    print("===> Number of word-tokens in the entire corpus: " + str(entire_num_word_tokens))


data_dir = load_dataset('BBC')
plot_distribution(data_dir.target, 350, 550)
processed_data = preprocess_dataset(data_dir.data)
x_train, x_test, y_train, y_test = split_dataset(processed_data, data_dir.target)
classifier = train_nb(x_train, y_train)
conf_matrix, report, acc_score, f1_macro_score, f1_weighted_score = test_nb(x_test, y_test, classifier)
prior0, prior1, prior2, prior3, prior4 = calculate_prior_prob(y_train)
size_vocabulary(x_train)
num_word_tokens(x_train)



### Output files for result ###
'''
with open('bbc-performance.txt', 'w') as file:
    file.write("---------- Multi-nomialNB default values, try 1 ----------s")
    file.write("Confusion Matrix: \n")
    file.write(str(conf_matrix))

    file.write("==========TRANING==========\n\n")
    file.write("Prediction label values: \n")
    file.write(str(train_prediction))
    file.write("\nActual label values: \n")
    file.write(str(train_labels))
    file.write('\n\n')
    file.write("Confusion Matrix: \n")
    file.write(str(conf_matrix_training))
    file.write('\n\n')
    file.write("Precision/Recall/f-1Measure: \n\n")
    file.write(report_training)
    file.write('\n\n')
    file.write("Accuracy Score: " + str(accuracy_training * 100) + "%")
    file.write('\n\n')
    file.write('Error Analysis:\nMisclassified samples: {}'.format(count_misclassified))
    file.write('\n\n\n')

    file.write("==========TESTING==========\n\n")
    file.write("Prediction label values: \n")
    file.write(str(eval_prediction))
    file.write("\nActual label values: \n")
    file.write(str(eval_labels))
    file.write('\n\n')
    file.write("Confusion Matrix: \n")
    file.write(str(conf_matrix_eval))
    file.write('\n\n')
    file.write("Precision/Recall/f-1Measure: \n\n")
    file.write(report_eval)
    file.write('\n\n')
    file.write("Accuracy Score: " + str(accuracy_eval * 100) + "%")
    file.write('\n\n')
    file.write('Error Analysis:\nMisclassified samples: {}'.format(count_misclassified))
    '''