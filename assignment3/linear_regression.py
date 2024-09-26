#Inshaad Merchant - 1001861293
import numpy as np

def get_phi_matrix(data, degree):
    
    X = data[:, :-1]  # Extract features from the data, assuming the last column is the target variable and it is excluded.
    rows, columns = X.shape # Get the number of samples (rows) and features (columns)

    total_columns = 1 + columns * degree # Calculate the total number of columns in the phi matrix, start with 1 for the intercept, and add 'degree' columns for each feature
    
    phi = np.ones((rows, total_columns)) # Pre-allocate the phi matrix with zeros (or ones for the first column)

    current_col = 1

    for c in range(columns): # Loop over each feature
        for d in range(1, degree + 1):  # Loop over each degree from 1 to 'degree'
            phi[:, current_col] = X[:, c] ** d  #multiply the current degree with the current feature and insert in the phi matrix.
            current_col += 1
    
    return phi

def linear_regression(training_file, test_file, degree, lambda_val):

    #TRAINING STAGE # Load the training and test datasets from the specified files
    training_data = np.loadtxt(training_file, dtype=np.float64)
    testing_data = np.loadtxt(test_file, dtype=np.float64)

    y_train = training_data[:, -1]
    y_test = testing_data[:, -1]

    # Compute phi for training and test data
    phi_trainingData = get_phi_matrix(training_data, degree)

    # Regularization matrix, excluding the intercept term
    I = np.identity(phi_trainingData.shape[1])
    I[0, 0] = 0

    # Compute weights using the regularized normal equation
    w = np.linalg.pinv(phi_trainingData.T.dot(phi_trainingData) + lambda_val * I).dot(phi_trainingData.T).dot(y_train)

    # Print weights
    print("Learned weights:")
    for index, weight in enumerate(w):
        print(f"w{index}={weight:.4f}")

    # TESTING STAGE
    
    phi_test = get_phi_matrix(testing_data, degree)
    predictions = phi_test @ w
    errors = predictions - y_test
    squared_errors = errors**2

    print("\nTesting results:")
    for i in range(len(testing_data)):
        print(f"ID={i+1}, output={predictions[i]:.4f}, target value={y_test[i]:.4f}, squared error={squared_errors[i]:.4f}")
