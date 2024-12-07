import numpy as np
import pandas as pd
from math import sqrt, exp, pi

def naive_bayes(training_file, test_file):
    # Read training data
    training_data = pd.read_csv(training_file, header=None, delim_whitespace=True)
    training_features = training_data.iloc[:, :-1]
    training_labels = training_data.iloc[:, -1]

    # Read test data
    test_features = pd.read_csv(test_file, header=None, delim_whitespace=True).iloc[:, :-1]
    test_labels = pd.read_csv(test_file, header=None, delim_whitespace=True).iloc[:, -1]

    # Calculate parameters within read_data for minimization
    classes = np.unique(training_labels)
    parameters = {}
    for a in classes:
        class_data = training_features[training_labels == a]
        parameters[a] = {
            'mean': class_data.mean(),
            'std': class_data.std().replace(0, sqrt(0.0001)).clip(lower=0.01),
            'prior': len(class_data) / len(training_labels)
        }

    # Print the training stage
    for cls in classes:
        for i, (mean, std) in enumerate(zip(parameters[cls]['mean'], parameters[cls]['std']), start=1):
            print(f"Class {cls}, attribute {i}, mean = {mean:.2f}, std = {std:.2f}")

    predictions = classify(parameters, test_features, test_labels)
    # Output results to the screen
    for prediction in predictions:
        index, predicted, probability, true_label = prediction
        accuracy = 1.0 if predicted == true_label else 0.0
        #print(f"ID={index}, predicted={predicted}, probability = {probability:.4f}, true={true_label}, accuracy={accuracy:.2f}")
    accuracy = np.mean([1 if pred[1] == true_label else 0 for pred in predictions])
    print(f"classification accuracy={accuracy:.4f}")

def gaussian_probability(x, mean, std):
    return (1 / (sqrt(2 * pi) * std)) * exp(-((x-mean)**2 / (2 * std**2)))

def classify(parameters, test_features, test_labels):
    results = []
    for index, x in test_features.iterrows():
        class_probs = {cls: np.log(parameters[cls]['prior']) for cls in parameters}
        for c, stats in parameters.items():
            for i in range(len(x)):
                class_probs[c] += np.log(gaussian_probability(x[i], stats['mean'][i], stats['std'][i]))
        best_class = max(class_probs, key=class_probs.get)
        probability = np.exp(max(class_probs.values()))
        results.append((index + 1, best_class, probability, test_labels.iloc[index]))
    return results