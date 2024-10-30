#Inshaad Merchant ----- UTA ID: 1001861293 ----- Assignment 5
import numpy as np
import pandas as pd

def entropy_calculation(labels):
    if len(labels) == 0:
        return 0
    unique_classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def info_gain_calc(data, labels, feature_idx, threshold):
    parent_entropy = entropy_calculation(labels)
    
    # Split the data
    left_mask = data[:, feature_idx] < threshold
    right_mask = ~left_mask
    
    # Calculate weighted entropy of children
    left_entropy = entropy_calculation(labels[left_mask])
    right_entropy = entropy_calculation(labels[right_mask])
    
    # Calculate weights
    total_samples = len(labels)
    left_weight = np.sum(left_mask) / total_samples
    right_weight = np.sum(right_mask) / total_samples
    
    # Calculate information gain
    weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
    information_gain = parent_entropy - weighted_entropy
    
    return information_gain

def find_best_split(data, labels, random_feature=None):
    best_gain = -1
    best_feature = -1
    best_threshold = -1
    
    # If random_feature is provided, only consider that feature
    features_to_try = [random_feature] if random_feature is not None else range(data.shape[1])
    
    for feature_idx in features_to_try:
        feature_values = data[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        # Try all possible thresholds
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            gain = info_gain_calc(data, labels, feature_idx, threshold)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

def build_tree(node_id, data, labels, min_samples, n_classes, is_random=False, tree_id=1, max_depth=10):

    def _build_tree_recursive(node_id, data, labels, min_samples, current_depth=0):
        # Initialize the node with consistent class dimensionality
        node = {
            'tree': tree_id,
            'node': node_id,
            'feature': -1,
            'threshold': -1,
            'gain': 0,
            'left': None,
            'right': None,
            'class_counts': np.bincount(labels.astype(int), minlength=n_classes),
            'is_leaf': False
        }
        
        # Check stopping criteria
        if (len(data) < min_samples or 
            len(np.unique(labels)) == 1 or 
            current_depth >= max_depth):
            node['is_leaf'] = True
            return node
        
        # Find the best split
        random_feature = np.random.randint(data.shape[1]) if is_random else None
        best_feature, best_threshold, best_gain = find_best_split(data, labels, random_feature)
        
        # If no good split found, return leaf node
        if best_gain <= 0.01:
            node['is_leaf'] = True
            return node
        
        # Split the data
        left_mask = data[:, best_feature] < best_threshold
        right_mask = ~left_mask
        
        # Ensure minimum samples in each split
        if np.sum(left_mask) < min_samples//2 or np.sum(right_mask) < min_samples//2:
            node['is_leaf'] = True
            return node
        
        # Update node information
        node['feature'] = best_feature
        node['threshold'] = best_threshold
        node['gain'] = best_gain
        
        # Build subtrees
        left_node_id = 2 * node_id
        right_node_id = 2 * node_id + 1
        
        node['left'] = _build_tree_recursive(left_node_id, 
                                           data[left_mask], 
                                           labels[left_mask], 
                                           min_samples, 
                                           current_depth + 1)
        node['right'] = _build_tree_recursive(right_node_id, 
                                            data[right_mask], 
                                            labels[right_mask], 
                                            min_samples, 
                                            current_depth + 1)
        
        return node
    
    return _build_tree_recursive(node_id, data, labels, min_samples)

def print_tree(node):
    print(f"tree={node['tree']}, node={node['node']}, feature={node['feature']}, "
          f"thr={node['threshold']:6.2f}, gain={node['gain']:f}")    
    if node['left']:
        print_tree(node['left'])
    if node['right']:
        print_tree(node['right'])

def classify_instance(tree, instance, n_classes):
    if tree is None:
        return np.zeros(n_classes)
    
    node = tree
    while not node['is_leaf']:
        if instance[node['feature']] < node['threshold']:
            node = node['left']
        else:
            node = node['right']
        
        if node is None:  # Safety check
            return np.zeros(n_classes)
    
    # Convert counts to probabilities with consistent dimensionality
    counts = np.array(node['class_counts'])
    if np.sum(counts) > 0:
        probabilities = counts / np.sum(counts)
    else:
        probabilities = np.zeros(n_classes)
    
    return probabilities


def decision_tree(training_file, test_file, option, pruning_thr):
    # Load training and test data using pandas
    training_data = pd.read_csv(training_file, delim_whitespace=True, header=None)
    testing_data = pd.read_csv(test_file, delim_whitespace=True, header=None)
    
    # Store category mappings from training data
    category_mappings = {}
    for col in training_data.select_dtypes(include=['object']):
        training_data[col] = training_data[col].astype('category')
        category_mappings[col] = training_data[col].cat.categories
        training_data[col] = training_data[col].cat.codes
        # Map test data using the same categories from training
        testing_data[col] = pd.Categorical(testing_data[col], 
                                      categories=category_mappings[col]).codes
    
    # Convert to numpy arrays
    training_attributes = training_data.iloc[:, :-1].values
    training_labels = training_data.iloc[:, -1].values
    test_attributes = testing_data.iloc[:, :-1].values
    testing_labels = testing_data.iloc[:, -1].values
    
    # Get number of unique classes from training data
    n_classes = len(np.unique(training_labels))
    
    # Initialize list to store trees
    trees = []
    
    # Build tree(s) based on option
    if option == "optimized":
        # Build single optimized tree
        tree = build_tree(1, training_attributes, training_labels, pruning_thr, n_classes)
        trees.append(tree)
        # Print the optimized tree
        print_tree(tree)
    else:
        # Build random forest
        n_trees = int(option)
        for i in range(n_trees):
            tree = build_tree(1, training_attributes, training_labels, pruning_thr, 
                            n_classes, is_random=True, tree_id=i+1)
            trees.append(tree)
            # Print all trees
            print_tree(tree)
    
    print("...................Model Training Complete......................")
    
    # Classification phase
    total_accuracy = 0
    n_test = len(testing_labels)
    
    # Classify each test instance
    for i in range(n_test):
        instance = test_attributes[i]
        combined_probability = np.zeros(n_classes)
        
        for tree in trees:
            probability = classify_instance(tree, instance, n_classes)
            if probability is not None:
                combined_probability += probability
        
        # Normalize probabilities
        if len(trees) > 0:
            combined_probability = combined_probability / len(trees)
        
        # Make prediction using highest probability class
        predicted_class = np.argmax(combined_probability)
        true_class = int(testing_labels[i])
        
        # Binary accuracy calculation
        accuracy = 1.0 if predicted_class == true_class else 0.0
        total_accuracy += accuracy
        
        print(f"ID={i+1}, predicted={predicted_class}, true={true_class}, accuracy={accuracy:.2f}")
    
    # Print overall classification accuracy
    classification_accuracy = total_accuracy / n_test
    print(f"classification accuracy={classification_accuracy:6.4f}")