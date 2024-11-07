#Inshaad Merchant -------- 1001861293
import numpy as np
from collections import Counter

def normalize_data(training_data, testing_data):
    
    mean = np.mean(training_data, axis=0)
    std = np.std(training_data, axis=0, ddof=1)  # Using N-1 for standard deviation
    
    # Handle zero standard deviation
    std[std == 0] = 1
    
    # Normalization Formula: F(x) = (x - mean) / std
    normalized_training_data = (training_data - mean) / std
    normalized_testing_data = (testing_data - mean) / std
    
    return normalized_training_data, normalized_testing_data   

def knn_classify(training_file, testing_file, k):
    # Read and parse the training and test files
    training_data = np.loadtxt(training_file)
    testing_data = np.loadtxt(testing_file)
    
    # Separate features and labels
    X_train = training_data[:, :-1]  # All columns except last
    y_train = training_data[:, -1]   # Last column
    X_test = testing_data[:, :-1]
    y_test = testing_data[:, -1]
    
    # Normalize each dimension separately
    X_train_normalized, X_test_normalized = normalize_data(X_train, X_test)
    
    # Initialize variables for overall accuracy calculation
    total_accuracy = 0
    num_test_samples = len(X_test)
    
    # Process each test sample
    for i in range(num_test_samples):
        test_sample = X_test_normalized[i]
        true_class = y_test[i]
        
        # Calculate distances to all training samples
        distance = np.sqrt(np.sum((X_train_normalized - test_sample) ** 2, axis=1))
        
        # Get k nearest neighbors
        nearest_indices = np.argsort(distance)[:k]
        nearest_classes = y_train[nearest_indices]
        
        # Count occurrences of each class among nearest neighbors
        class_count = Counter(nearest_classes)
        max_count = max(class_count.values())
        
        # Find classes that tied for the highest count
        tied_classes = [cls for cls, count in class_count.items() if count == max_count]
        
        # Randomly choose from tied classes
        predicted_class = np.random.choice(tied_classes)
        
        # Calculate accuracy for this sample
        if len(tied_classes) == 1:  # No ties
            accuracy = 1.0 if predicted_class == true_class else 0.0
        else:  # Ties
            if true_class in tied_classes:
                accuracy = 1.0 / len(tied_classes)
            else:
                accuracy = 0.0
        
        # Add to total accuracy
        total_accuracy += accuracy
        
        # Print result for this test object
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' % 
              (i + 1, str(predicted_class), str(true_class), accuracy))
    
    # Calculate and print overall classification accuracy
    classification_accuracy = (total_accuracy / num_test_samples)
    print('classification accuracy=%6.4f' % classification_accuracy)