import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB

def read_documents(csv_file):
    try: 
        df = pd.read_csv(csv_file)
        print("Original Dataframe: \n" + str(df.sample(5)) + "\n\n")
    except:
        print("File not found.")
        sys.exit(0)
    return df


def plot_distribution(df):
    # Count the frequency of different Drug classes and plot them
    df['Drug'].value_counts().plot(kind='bar')
    plt.ylabel('Classes')
    plt.xlabel("Frequency")
    plt.title('Distribution of Class Drug')
    # Save graph
    plt.savefig('drug-distribution.pdf')
    # Show graph
    plt.show()


def convert_numerical(df):
    df_dummies = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Sex', 'Drug'])
    # F, M, drugA, drugB, drugC, drugX, drugY
    df_dummies.BP = pd.Categorical(df_dummies.BP, ordered=True, categories=['LOW', 'NORMAL', 'HIGH']).codes
    # LOW:0, NORMAL:1, HIGH:2
    df_dummies.Cholesterol = pd.Categorical(df_dummies.Cholesterol, ordered=True, categories=['NORMAL', 'HIGH']).codes
    # NORMAL:0, HIGH:1
    print("Updated Dataframe after numerical conversion: \n" + str(df_dummies.sample(5)) + "\n\n")
    return df_dummies




dataset = read_documents('drug200.csv')
plot_distribution(dataset)
dataset = convert_numerical(dataset)

