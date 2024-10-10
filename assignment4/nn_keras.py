import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def nn_keras(directory, dataset, layers, units_per_layer, epochs):
    # Construct file paths
    training_file = f"{directory}/{dataset}_training.txt"
    test_file = f"{directory}/{dataset}_test.txt"
    
    # Load data
    training_data = pd.read_csv(training_file, header=None, sep=r'\s+')
    testing_data = pd.read_csv(test_file, header=None, sep=r'\s+')

    # Separate features and labels
    X_train = training_data.iloc[:, :-1].values.astype(float)
    y_train = training_data.iloc[:, -1].values
    X_test = testing_data.iloc[:, :-1].values.astype(float)
    y_test = testing_data.iloc[:, -1].values
    
    # Find the maximum absolute value for normalization
    max_abs_value = np.max(np.abs(X_train))
    
    # Normalize the data
    X_train /= max_abs_value
    X_test /= max_abs_value
    
     # Convert labels from strings to numeric indices
    labels = np.unique(np.concatenate([y_train, y_test]))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_train = np.array([label_to_idx[label] for label in y_train])
    y_test = np.array([label_to_idx[label] for label in y_test])

    # Create the model
    model = Sequential([
        # Input shape is specified in the first Dense layer
        Dense(len(labels), activation='sigmoid', input_shape=(X_train.shape[1],))
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    
    model.fit(X_train, y_train, epochs=epochs, verbose=1)
    # Get predictions and calculate individual accuracies
    probabilities = model.predict(X_test)
    predicted_classes = np.argmax(probabilities, axis=1)
    accuracies = []
    
    for i in range(len(y_test)):
        true_class = y_test[i]
        predicted_probabilities = probabilities[i]
        top_prediction = np.max(predicted_probabilities)
        
        # Handling ties
        tied_classes = np.where(predicted_probabilities == top_prediction)[0]
        if len(tied_classes) == 1:
            if true_class in tied_classes:
                accuracy = 1.0
            else:
                accuracy = 0.0
        else:
            if true_class in tied_classes:
                accuracy = 1 / len(tied_classes)
            else:
                accuracy = 0.0
        
        accuracies.append(accuracy)
        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (i, labels[predicted_classes[i]], labels[true_class], accuracy))
    
    # Calculate and print overall classification accuracy
    classification_accuracy = np.mean(accuracies)
    print('classification accuracy=%6.4f%%\n' % (classification_accuracy*100))
