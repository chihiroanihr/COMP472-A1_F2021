import sys
from os import path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB

def read_documents(csv_file):
    try: 
        dataset = pd.read_csv(csv_file)
        print(dataset.sample(5))
    except:
        print("File not found.")
        sys.exit(0)


read_documents('drug200.csv')

