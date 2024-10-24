# FIRST: RENAME THIS FILE TO sentiment_utils.py 

# YOUR NAMES HERE:


"""

CS 4120
Homework 4
Fall 2024

Utility functions for HW 4, to be imported into the corresponding notebook(s).

Add any functions to this file that you think will be useful to you in multiple notebooks.
"""
# fancy data structures
from collections import defaultdict, Counter
# for tokenizing and precision, recall, f_measure, and accuracy functions
import nltk
from nltk.metrics import precision, recall, f_measure, accuracy
# for plotting
import matplotlib.pyplot as plt
# so that we can indicate a function in a type hint
from typing import Callable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

nltk.download('punkt')

def generate_tuples_from_file(training_file_path: str) -> list:
    """
    Generates data from file formated like:

    tokenized text from file: [[word1, word2, ...], [word1, word2, ...], ...]
    labels: [0, 1, 0, 1, ...]
    
    Parameters:
        training_file_path - str path to file to read in
    Return:
        a list of lists of tokens and a list of int labels
    """
    # PROVIDED
    f = open(training_file_path, "r", encoding="utf8")
    X = []
    y = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.strip().split("\t")
        if len(dataInReview) != 3:
            continue
        else:
            t = tuple(dataInReview)
            if (not t[2] == '0') and (not t[2] == '1'):
                print("WARNING")
                continue
            X.append(nltk.word_tokenize(t[1]))
            y.append(int(t[2]))
    f.close()  
    return X, y


"""
NOTE: for all of the following functions, we have provided the function signature and docstring, *that we used*, as a guide.
You are welcome to implement these functions as they are, change their function signatures as needed, or not use them at all.
Make sure that you properly update any docstrings as needed.
"""


def get_prfa(dev_y: list, preds: list, verbose=False) -> tuple:
    """
    Calculate precision, recall, f1, and accuracy for a given set of predictions and labels.
    Args:
        dev_y: list of true labels
        preds: list of predictions
        verbose: whether to print the metrics
    Returns:
        tuple of precision, recall, f1, and accuracy
    """
    refsets = {}
    testsets = {}

    # Loop through true labels and predictions
    for i in range(len(dev_y)):
        label = dev_y[i]
        pred = preds[i]
        
        # Initialize sets for refsets and testsets if necessary
        if label not in refsets:
            refsets[label] = set()
        if pred not in testsets:
            testsets[pred] = set()
        
        # Add index i to the correct label in refsets and testsets
        refsets[label].add(i)
        testsets[pred].add(i)

    # Initialize cumulative metrics
    precision_score = recall_score = f1_score = 0.0
    label_count = len(refsets)

    # Calculate precision, recall, f1 for each label
    for label in refsets:
        if label in testsets:
            precision_score += precision(refsets[label], testsets[label]) or 0.0
            recall_score += recall(refsets[label], testsets[label]) or 0.0
            f1_score += f_measure(refsets[label], testsets[label]) or 0.0

    # Calculate average precision, recall, and f1 over all labels
    precision_score /= label_count
    recall_score /= label_count
    f1_score /= label_count

    # Calculate accuracy
    accuracy_score = accuracy(dev_y, preds)

    if verbose:
        print(f"Precision: {precision_score}")
        print(f"Recall: {recall_score}")
        print(f"F1 Score: {f1_score}")
        print(f"Accuracy: {accuracy_score}")

    return precision_score, recall_score, f1_score, accuracy_score

def create_training_graph(metrics_fun: Callable, train_feats: list, dev_feats: list, kind: str, savepath: str = None, verbose: bool = False) -> None:
    """
    Create a graph of the classifier's performance on the dev set as a function of the amount of training data.
    Args:
        metrics_fun: a function that takes in training data and dev data and returns a tuple of metrics
        train_feats: a list of training data in the format [(feats, label), ...]
        dev_feats: a list of dev data in the format [(feats, label), ...]
        kind: the kind of model being used (will go in the title)
        savepath: the path to save the graph to (if None, the graph will not be saved)
        verbose: whether to print the metrics
    """
    # Data percentages to evaluate the model with
    data_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Lists to store metrics for each percentage
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    # Loop through different percentages of the training data
    for percent in data_percentages:
        # Determine the subset of training data to use
        data_size = int(len(train_feats) * (percent / 100))
        current_train_data = train_feats[:data_size]
        
        # Call the provided metrics function to get precision, recall, f1, and accuracy
        precision_score, recall_score, f1_score, accuracy_score = metrics_fun(current_train_data, dev_feats)
        
        # Append the results to the respective lists
        precisions.append(precision_score)
        recalls.append(recall_score)
        f1_scores.append(f1_score)
        accuracies.append(accuracy_score)
        
        # Optionally print the metrics for each percentage of data
        if verbose:
            print(f"Training data percentage: {percent}%")
            print(f"Precision: {precision_score:.4f}, Recall: {recall_score:.4f}, F1 Score: {f1_score:.4f}, Accuracy: {accuracy_score:.4f}\n")
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    
    # Plot precision, recall, f1, and accuracy
    plt.plot(data_percentages, precisions, label='Precision', marker='o')
    plt.plot(data_percentages, recalls, label='Recall', marker='o')
    plt.plot(data_percentages, f1_scores, label='F1 Score', marker='o')
    plt.plot(data_percentages, accuracies, label='Accuracy', marker='o')
    
    # Add title and labels
    plt.title(f'{kind} Classifier Performance on Dev Set as a Function of Training Data')
    plt.xlabel('Training Data Percentage (%)')
    plt.ylabel('Performance')
    plt.legend(loc='best')
    
    # Add grid for better readability
    plt.grid(True)
    
    # Save the graph if a savepath is provided
    if savepath:
        plt.savefig(savepath)
        if verbose:
            print(f"Graph saved to {savepath}")
    
    # Show the plot
    plt.show()
    pass

# Helper function to train and evaluate the Naive Bayes classifier
def nb_metrics_fun(train_feats, dev_feats):
    """
    Train a Naive Bayes classifier and evaluate it on the dev set.
    
    Args:
        train_feats: List of training data [(features, label), ...]
        dev_feats: List of dev data [(features, label), ...]
        
    Returns:
        tuple: (precision, recall, f1, accuracy)
    """
    # Train the Naive Bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(train_feats)
    
    # Get the dev data features and true labels
    dev_data = [feats for feats, label in dev_feats]
    dev_labels = [label for feats, label in dev_feats]
    
    # Make predictions on the dev set
    dev_preds = [classifier.classify(feats) for feats in dev_data]
    
    # Evaluate the model using the get_prfa helper function
    precision_score, recall_score, f1_score, accuracy_score = get_prfa(dev_labels, dev_preds)
    
    return precision_score, recall_score, f1_score, accuracy_score
def logistic_regression_metrics_fun(train_feats, dev_feats):
    """
    Train a Logistic Regression model and evaluate it on the dev set.
    
    Args:
        train_feats: List of training data [(features, label), ...]
        dev_feats: List of dev data [(features, label), ...]
        
    Returns:
        tuple: (precision, recall, f1, accuracy)
    """
    # Extract features and labels from the training data
    X_train = [feats for feats, label in train_feats]
    y_train = [label for feats, label in train_feats]
    
    # Extract features and labels from the dev data
    X_dev = [feats for feats, label in dev_feats]
    y_dev = [label for feats, label in dev_feats]
    
    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=200)  # You can adjust max_iter as needed
    model.fit(X_train, y_train)
    
    # Make predictions on the dev set
    y_pred = model.predict(X_dev)
    
    # Calculate precision, recall, f1, and accuracy
    precision = precision_score(y_dev, y_pred, average='binary')
    recall = recall_score(y_dev, y_pred, average='binary')
    f1 = f1_score(y_dev, y_pred, average='binary')
    accuracy = accuracy_score(y_dev, y_pred)
    
    return precision, recall, f1, accuracy
def create_index(all_train_data_X: list) -> list:
    """
    Given the training data, create a list of all the words in the training data.
    Args:
        all_train_data_X: a list of all the training data in the format [[word1, word2, ...], ...]
    Returns:
        vocab: a list of all the unique words in the training data
    """
    # figure out what our vocab is and what words correspond to what indices
    #TODO: implement this function
    pass


def featurize(vocab: list, data_to_be_featurized_X: list, binary: bool = False, verbose: bool = False) -> list:
    """
    Create vectorized BoW representations of the given data.
    Args:
        vocab: a list of words in the vocabulary
        data_to_be_featurized_X: a list of data to be featurized in the format [[word1, word2, ...], ...]
        binary: whether or not to use binary features
        verbose: boolean for whether or not to print out progress
    Returns:
        a list of sparse vector representations of the data in the format [[count1, count2, ...], ...]
    """
    # using a Counter is essential to having this not take forever
    #TODO: implement this function
    pass