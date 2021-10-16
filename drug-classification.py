
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB

def read_documents(csv_file):
    try: 
        print("a")
    except:
        print("File not found.")
        sys.exit(0)



