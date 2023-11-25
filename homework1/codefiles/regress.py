import numpy as np

X_seen = np.load('X_seen.npy', allow_pickle=True, encoding='bytes') #X_seen.shape = (40,)
class_attributes_seen = np.load('class_attributes_seen.npy', allow_pickle=True, encoding='bytes') #class_attributes_seen.shape = (40, 85)
class_attributes_unseen = np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='bytes') #class_attributes_unseen.shape = (10, 85)

Xtest = np.load('Xtest.npy', allow_pickle=True, encoding='bytes') #Xtest.shape = (6180, 4096)
Ytest = np.load('Ytest.npy', allow_pickle=True, encoding='bytes') #Ytest.shape = (6180, 1)
Ytest -= 1 # argmin returns from (0,,9) but Ytest has values between (1, 10)

NUM_FEATURES = Xtest.shape[1]
NUM_SEEN_CLASSES, NUM_CLASS_ATTRIBUTE_FEATURES = class_attributes_seen.shape
NUM_UNSEEN_CLASSES = class_attributes_unseen.shape[0]
NUM_TEST_EXAMPLES = Xtest.shape[0]

mu_seen = np.zeros((NUM_SEEN_CLASSES, NUM_FEATURES)) #No of seen classes * No of features
mu_unseen = np.zeros((NUM_UNSEEN_CLASSES, NUM_FEATURES)) #No of unseen classes * No of features
Y_pred = np.zeros(NUM_TEST_EXAMPLES, dtype='int')

for i in range(NUM_SEEN_CLASSES):
    mu_seen[i] = np.mean(X_seen[i], axis=0) # Step 1: Computing the means of each class

hyperparam = [0.01, 0.1, 1, 10, 20, 50, 100] # The list of lambda values to be tried with the model

for param in hyperparam:
    temp_mat = np.matmul(class_attributes_seen.T, class_attributes_seen) + param * np.eye(NUM_CLASS_ATTRIBUTE_FEATURES) # Temporary matrix for calculating As^T As + lambda I
    temp_mat = np.linalg.inv(temp_mat) # Temporarily storing the inverse
    W = np.matmul(np.matmul(temp_mat, class_attributes_seen.T), mu_seen) # Step 2: Learning the regression model
    
    # print(W.shape)
    # print(class_attributes_unseen.shape)
    
    mu_unseen = np.matmul(W.T, class_attributes_unseen.T).T # Step 3: Computing the mean of the unseen classes and taking the transpose for compatability in multiplication
    distances = np.linalg.norm(np.expand_dims(Xtest, axis=1) - mu_unseen, axis=2) # Applying the model to compute distances from each class mean
    unseen_class_predictions = np.argmin(distances, axis=1) # Step 4: Using the model to compute labels

    unseen_class_predictions = np.reshape(unseen_class_predictions, (Xtest.shape[0], 1)) # Reshaping the predictions for accuracy calculation since Ytest has size (6180,1)
    correct_predictions = (unseen_class_predictions == Ytest).astype(np.float32)
    accuracy = np.mean(correct_predictions) * 100.0 # Step 5: Computing the accuracy of classification
    print(param, accuracy)