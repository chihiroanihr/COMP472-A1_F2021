import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import f1_score


def read_documents(csv_file):
    try: 
        df = pd.read_csv(csv_file)
        print("<< Original Dataframe >>\n" + str(df.sample(5)) + "\n\n")
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
    # Get dummy classes: F, M, drugA, drugB, drugC, drugX, drugY
    df_dummies = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Sex'])
    # Convert categorical BP class into numerical:  LOW:0, NORMAL:1, HIGH:2
    df_dummies.BP = pd.Categorical(df_dummies.BP, ordered=True, categories=['LOW', 'NORMAL', 'HIGH']).codes
    # Convert categorical Cholesterol class into numerical:  NORMAL:0, HIGH:1
    df_dummies.Cholesterol = pd.Categorical(df_dummies.Cholesterol, ordered=True, categories=['NORMAL', 'HIGH']).codes
    # Convert categorical Drug class into numerical:  drugA: 0, drugB: 1, drugC: 2, drugX: 3, drugY: 4
    df_dummies.Drug =  pd.Categorical(df_dummies.Drug, ordered=False, categories=['drugA', 'drugB', 'drugC', 'drugX', 'drugY']).codes
    print("<< Updated Dataframe after numerical conversion >>\n" + str(df_dummies.sample(5)) + "\n\n")
    return df_dummies


def split_dataset(df):
    x_df = df.drop('Drug', axis=1)
    y_df = df['Drug']
    print("<< Dataset X(Data) >>\n" + str(x_df) + "\n")
    print("<< Dataset Y(target) >>\n" + str(y_df))
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=None)  # no shuffling before dataset split
    print("\n\n===> Dataset splitted")
    print("     x_train size: " + str(x_train.shape))
    print("     y_train size: " + str(y_train.shape))
    print("     x_test size: " + str(x_test.shape))
    print("     y_test size: " + str(y_test.shape))
    return x_train, x_test, y_train, y_test


def train_nb(x_train, y_train, smoothing=1.0): # Default smoothing value: 1
    # Create classifier model list
    gaussianNB =  GaussianNB()
    baseDT = DecisionTreeClassifier()
    topDT = DecisionTreeClassifier()
    perceptron = Perceptron()
    baseMLP = MLPClassifier()
    topMLP = MLPClassifier()
    clfmodel_list = [gaussianNB, baseDT, topDT, perceptron, baseMLP, topMLP]
    # Create parameters list for certain classifier which requires them
    paramsTopDT = {'criterion': ['gini', 'entropy'], 'max_depth': [3,6], 'min_samples_leaf': [0.02,1,10]}
    paramsBaseMLP = {'activation': ['logistic'], 'solver': ['sgd']}
    paramsTopMLP = {'activation': ['logistic', 'tanh', 'relu', 'identity'], 'hidden_layer_sizes': [(30, 50), (10, 10, 10)], 'solver': ['sgd', 'adam']}
    parameters_list = [{}, {}, paramsTopDT, {}, paramsBaseMLP, paramsTopMLP] # 0th, 1th, 3th classifier models uses default parameters thus {}

    results=[]

    for i in range(len(clfmodel_list)):
        grid=GridSearchCV(estimator=clfmodel_list[i], param_grid=parameters_list[i])
        grid.fit(x_train, y_train) # train it!
            #storing result
        results.append\
        (
            {
                'grid': grid,
                'classifier': grid.best_estimator_,
                'best score': grid.best_score_,
                'best params': grid.best_params_,
                'cv': grid.cv
            }
        )

    for result in results:
        print(result)
        print("\n")

    return x_train, y_train, grid


def test_nb(x_test, y_test, grid):
    x_test_pred = grid.predict(x_test)
    conf_matrix = confusion_matrix(y_test, x_test_pred)
    report = classification_report(y_test, x_test_pred)
    acc_score = round(accuracy_score(y_test, x_test_pred), 4)
    f1_macro_score = round(f1_score(y_test, x_test_pred, average='macro'), 4)
    f1_weighted_score = round(f1_score(y_test, x_test_pred, average='weighted'), 4)
    print("\n\n << " + str(grid) + "\n")
    print("===> Result Prediction (Actual value & Predicted value):\n")
    print("---> Confusion matrix: \n" + str(conf_matrix) + "\n")
    print("---> Classification report (Precision, Recall, F1): \n\n" + str(report))
    print("---> Accuracy score: " + str(acc_score) + " (" + str(round(acc_score*100, 2)) + "%)")
    print("---> F1-score with macro average: " + str(f1_macro_score) + " (" + str(round(f1_macro_score*100, 2)) + "%)")
    print("---> F1-score with weighted average: " + str(f1_weighted_score) + " (" + str(round(f1_weighted_score*100, 2)) + "%)")
    return conf_matrix, report, acc_score, f1_macro_score, f1_weighted_score

dataset = read_documents('drug200.csv')
plot_distribution(dataset)
dataset = convert_numerical(dataset)
x_train, x_test, y_train, y_test = split_dataset(dataset)
x_train, y_train, grid = train_nb(x_train, y_train)
conf_matrix, report, acc_score, f1_macro_score, f1_weighted_score = test_nb(x_test, y_test, grid)

