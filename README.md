# COMP472-A1_Fall-2021: Experiments with Machine Learning (Assignment)

## Introduction
This project is part of the COMP472 course on Artificial Intelligence, focusing on experimenting with different machine learning algorithms and datasets. The primary objective is to gain practical experience with text classification and drug classification tasks using various classifiers.

## Project Scope
- **Duration**: Fall 2021 semester (Deadline: October 18, 2021)
- **Resources**: Python 3.8, scikit-learn library, matplotlib, pandas
- **Key Activities**: Data preprocessing, model training, performance evaluation, and result analysis

## Tasks and Implementations

### Task 1: Text Classification
- Dataset: BBC news articles (2225 documents, 5 classes)
- Classifier: Multinomial Naive Bayes
- Key Steps:
  1. Load and visualize dataset distribution
  2. Preprocess text data
  3. Split dataset (80% training, 20% testing)
  4. Train and evaluate models with different smoothing values
  5. Analyze and report various metrics

### Task 2: Drug Classification
- Dataset: Drug dataset (categorical and numerical features)
- Classifiers: 
  - Gaussian Naive Bayes
  - Decision Tree (Base and Optimized)
  - Perceptron
  - Multi-Layer Perceptron (Base and Optimized)
- Key Steps:
  1. Load and preprocess data
  2. Visualize class distribution
  3. Train and evaluate multiple classifiers
  4. Perform grid search for hyperparameter tuning
  5. Analyze and compare model performances

## Outcomes and Results

### Text Classification
- Implemented Multinomial Naive Bayes with different smoothing values
- Achieved high accuracy (98.2%) and F1-scores
- Analyzed word frequencies, zero-frequency words, and log probabilities

### Drug Classification
- Implemented and compared 6 different classifiers
- Decision Tree models showed the best performance (100% accuracy)
- MLP and Gaussian NB showed moderate performance
- Perceptron showed the lowest performance

## Challenges and Resolutions
- Handling imbalanced datasets
- Implementing various classifiers and understanding their parameters
- Resolving these challenges through careful data preprocessing and parameter tuning

## Lessons Learned
- Importance of data preprocessing in machine learning tasks
- Impact of smoothing values on Naive Bayes performance
- Effectiveness of decision trees for categorical data
- Significance of hyperparameter tuning for model optimization

## Conclusion
This project provided hands-on experience with real-world machine learning tasks, emphasizing the importance of proper data handling, model selection, and performance evaluation in AI applications.

## Files in the Repository
- [`COMP472_A1_Instruction.pdf`](/COMP472_A1_Instruction.pdf): Instruction of this assignment distributed by COMP472 Professor.
- [`text-classification.py`](/text-classification.py): Implementation of BBC news classification
- [`drug-classification.py`](/drug-classification.py): Implementation of drug classification
- [`bbc-performance.txt`](/bbc-performance.txt): Performance metrics for text classification
- [`drugs-performance.txt`](/drugs-performance.txt): Performance metrics for drug classification
- [`BBC-distribution.pdf`](/BBC-distribution.pdf): Visualization of BBC dataset distribution
- [`drug-distribution.pdf`](/drug-distribution.pdf): Visualization of drug dataset distribution
- [`bbc-discussion.txt`](/bbc-discussion.txt): Analysis of text classification results
- [`drugs-discussion.txt`](/drugs-discussion.txt): Analysis of drug classification results
- [`COMP472_A1_Presentation.pdf`](/COMP472_A1_Presentation.pdf): Detailed presentation of project results and analysis
- [`requirements.txt`](/requirements.txt): List of required Python packages

## Presentation
The [`COMP472_A1_Presentation.pdf`](/COMP472_A1_Presentation.pdf) file contains a comprehensive overview of the project results and analysis. It includes:
- Visualizations of dataset distributions
- Detailed results for both text and drug classification tasks
- Performance comparisons of different classifiers
- Analysis of model behaviors with different parameters
- Key findings and insights from the experiments
This presentation serves as a valuable resource for understanding the project outcomes in depth and provides visual representations of the results.

## How to Run
1. Ensure Python 3.8 and required packages are installed:
   ```
   pip install -r requirements.txt
   ```
2. For text classification:
   ```
   python text-classification.py
   ```
3. For drug classification:
   ```
   python drug-classification.py
   ```

## Note
This project is for educational purposes as part of the COMP472 course. The BBC dataset and drug dataset are provided for academic use only.