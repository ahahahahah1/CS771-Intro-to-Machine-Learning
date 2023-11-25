import numpy as np

X_seen = np.load('X_seen.npy', allow_pickle=True, encoding='bytes') #X_seen.shape = (40,)
class_attributes_seen = np.load('class_attributes_seen.npy', allow_pickle=True, encoding='bytes') #class_attributes_seen.shape = (40, 85)
class_attributes_unseen = np.load('class_attributes_unseen.npy', allow_pickle=True, encoding='bytes') #class_attributes_unseen.shape = (10, 85)

Xtest = np.load('Xtest.npy', allow_pickle=True, encoding='bytes') #Xtest.shape = (6180, 4096)
Ytest = np.load('Ytest.npy', allow_pickle=True, encoding='bytes') #Ytest.shape = (6180, 1)

NUM_FEATURES = Xtest.shape[1]
NUM_SEEN_CLASSES = class_attributes_seen.shape[0]
NUM_UNSEEN_CLASSES = class_attributes_unseen.shape[0]
NUM_TEST_EXAMPLES = Xtest.shape[0]

mu_seen = np.zeros((NUM_SEEN_CLASSES, NUM_FEATURES)) #No of seen classes * No of features
similarity = np.zeros((NUM_UNSEEN_CLASSES,NUM_SEEN_CLASSES), dtype=float) #No of unseen classes * No of seen classes (each entry of this matrix stores the similarity between a seen and unseen class)
mu_unseen = np.zeros((NUM_UNSEEN_CLASSES, NUM_FEATURES)) #No of unseen classes * No of features
Y_pred = np.zeros(NUM_TEST_EXAMPLES, dtype='int')

for i in range(NUM_SEEN_CLASSES):
    mu_seen[i] = np.mean(X_seen[i], axis=0) # Step 1: Computing the means of each class

similarity = np.dot(class_attributes_unseen, class_attributes_seen.T) # Step 2: Computing the similarity using dot product
similarity = similarity / np.sum(similarity, axis=1, keepdims=1) # Step 3: Normalizing the similarity vectors (along each row since row-wise sum must be 1

similarity = similarity[:, :, np.newaxis] # Creating a new axis since we would require it for multiplication
mu_unseen = np.sum(similarity * mu_seen, axis=1) # Step 4: Computing means for unseen classes 
# print(mu_unseen.shape)

distances = np.linalg.norm(np.expand_dims(Xtest, axis=1) - mu_unseen, axis=2) # Applying the model to compute distances from each class mean
unseen_class_predictions = np.argmin(distances, axis=1) # Step 5: Using the model to compute labels

unseen_class_predictions = np.reshape(unseen_class_predictions, (Xtest.shape[0], 1)) # Reshaping the predictions for accuracy calculation since Ytest has size (6180,1)
Ytest -= 1 # argmin returns from (0,,9) but Ytest has values between (1, 10)
correct_predictions = (unseen_class_predictions == Ytest).astype(np.float32)
accuracy = np.mean(correct_predictions) * 100.0 # Step 6: Computing the accuracy of classification
print(accuracy)