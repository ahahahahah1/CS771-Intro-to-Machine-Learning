import numpy as np
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt

def read_from_file(file_name):
    # Needs the file to be present in the same directory
    """
    args: file_name: the name of the file to be read
    outputs: input_list, output_list : 2 lists containing the inputs and outputs"""
    input_list = []
    output_list = []
    
    with open(file_name, 'r') as file:
        for line in file:
            input, output = map(float, line.strip().split())
            input_list.append(input)
            output_list.append(output)

    return np.array(input_list), np.array(output_list)


def rbf_kernel(x1, x2, gamma):
    """
    Compute the RBF (Gaussian) kernel between X1 and X2.
    """
    return np.exp(-gamma * (norm(x1 - x2) ** 2))

def transform_input(X, landmarks, gamma=0.1):
    num_samples = X.shape[0]
    num_landmarks = landmarks.shape[0]
    transformation = np.zeros((num_samples, num_landmarks))
    for i in range(num_samples):
        for j in range(num_landmarks):
            transformation[i,j] = rbf_kernel(X[i], landmarks[j], gamma)
    return transformation

def kernel_ridge_regression(X_train, y_train, gamma=0.1, lambd=1.0):
    """
    Perform kernel ridge regression.
    
    :param gamma: Kernel coefficient for the RBF kernel.
    :param lambd: Regularization parameter.
    """
    n_samples = X_train.shape[0]
    
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = rbf_kernel(X_train[i], X_train[j], gamma) # computing the similarity btw x_i x_j
    
    I = np.eye(n_samples)
    K_train_reg = K + lambd * I
    alpha_vector = inv(K_train_reg).dot(y_train)

    return alpha_vector

def predict_kernel_ridge(predictor, X_test, X_train, gamma=0.1):
    K_test = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            K_test[i, j] = rbf_kernel(X_test[i], X_train[j], gamma)
    predictions = K_test.dot(predictor)
    
    return predictions

def landmark_ridge(X_train, Y_train, X_test, L, gamma=0.1, lambd=0.1):
    landmarks = np.random.choice(X_train, L, replace=False)
    train_examples_transformed = transform_input(X_train, landmarks, gamma)
    test_examples_transformed = transform_input(X_test, landmarks, gamma)
    
    num_samples, num_landmarks = train_examples_transformed.shape
    I = np.eye(num_landmarks)
    X_squared = train_examples_transformed.T @ train_examples_transformed
    weights = inv(X_squared + lambd * I) @ train_examples_transformed.T @ Y_train
    return test_examples_transformed @ weights

if __name__ == '__main__':
    X_train, Y_train = read_from_file('ridgetrain.txt')
    print(X_train.shape)
    X_test, Y_test = read_from_file('ridgetest.txt')
    lambd_vals = [0.1, 1, 10, 100]
    L_vals = [2, 5, 20, 50, 100]

    # Kernel ridge regression
    plt.figure(figsize=(12,10))
    for i, lambd in enumerate(lambd_vals):
        weights = kernel_ridge_regression(X_train, Y_train, 0.1, lambd)
        y_pred = predict_kernel_ridge(weights, X_test, X_train)
        RMSE = np.sqrt(mean_squared_error(y_pred, Y_test))
        print(RMSE)
        plt.subplot(2, 2, i + 1)
        plt.scatter(X_test, Y_test, color='blue')
        plt.scatter(X_test, y_pred, color='red')
        plt.xlabel("Input value")
        plt.ylabel("Output value")
        plt.title("Performance of Kernel Ridge Regression for lambda= " + str(lambd), fontsize=10)
    # plt.show()
    plt.savefig('part1_1.png')
    
    # Landmark ridge regression
    for i, L in enumerate(L_vals):
        for lambd in lambd_vals:
            y_pred = landmark_ridge(X_train, Y_train, X_test, L, 0.1, lambd)
            RMSE = np.sqrt(mean_squared_error(y_pred, Y_test))
            print(f"RMSE for lambda = {lambd} and L = {L} is {RMSE}")
            if(lambd == 0.1):
                plt.figure()
                plt.scatter(X_test, Y_test, color='blue')
                plt.scatter(X_test, y_pred, color='red')
                plt.xlabel("Input value")
                plt.ylabel("Output value")
                plt.title("Performance of Kernel Ridge Regression for L= " + str(L), fontsize=10)
                plt.savefig(f"part1_2_{L}.png")
    
    