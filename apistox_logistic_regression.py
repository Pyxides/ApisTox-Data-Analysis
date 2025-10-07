import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pandas.plotting import parallel_coordinates

'''
Carlos Fuentes
Python Version: 3.12.7
Data: dataset_final.csv from apistox 
Source: UCI Machine Learning Repository
Link: https://archive.ics.uci.edu/dataset/995/apistox
Download link(zip): https://archive.ics.uci.edu/static/public/995/apistox.zip
Binary Classification task: Predict 'label' value, 1 = toxic, 0 = non-toxic
Features(binary) use to predict: 'herbicide', 'insecticide', 'fungicide', and 'other_agrochemical'
Model: Logistic Regression
Decision rule: y = 1 for h(x) >= 0.5, y = 0 for h(x) < 0.5
'''

csv = pd.read_csv('dataset_final.csv')
# print(csv.head())

# Gradient computation
def compute_gradients(x, y, w, w_0):
    m = len(y)
    dw = np.zeros_like(w)
    dw_0 = 0

    for i in range(m):
        z = np.dot(x[i], w) + w_0 #linear combo of feat.
        h = sigmoid(z) #prediction
        error = h -y[i]
        dw += error * x[i] #gradient of weights
        dw_0 += error #gradient of bias
    return dw / m, dw_0 / m

# Gradient descent implementation
def gradient_descent(x, y, w, w_0, alpha, num_iterations):
    predictions_raw = [] # stores raw predictions of ea. iteration for scatterplot
    for i in range(num_iterations):
        dw, dw_0 = compute_gradients(x, y, w, w_0)
        w -= alpha * dw #update weights
        w_0 -= alpha * dw_0 #update bias

        # Store raw predictions for scatterplot
        z = np.dot(x, w) + w_0

        #append sigmoid value of each prediction
        predictions_raw.append(sigmoid(z))

        # Print cost every 10 iterations
        if i % 100 == 0:
            cost = compute_cost(x, y, w, w_0)
            print(f"Iteration {i}: Cost = {cost:.4f}, Weights = {w}, Bias(w_0) = {w_0:.4f}")

    return w, w_0, predictions_raw

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression cost function
# getting divide by zero and invalid value erros for cost += -y[i] * np.log(h) - (1 - y[i]) * np.log(1 - h)
def compute_cost(x, y, w, w_0):
    m = len(y)
    cost = 0
    for i in range(m):
        z = np.dot(x[i], w) + w_0
        h = sigmoid(z)
        epsilon = 1e-10 #avoid division by zero
        cost += -y[i] * np.log(h + epsilon) - (1 - y[i]) * np.log(1 - h + epsilon)
    return cost / m

#x = csv[['herbicide', 'insecticide', 'fungicide', 'other_agrochemical']]
x = csv[['herbicide', 'insecticide', 'fungicide', 'other_agrochemical']].values
y = csv['label'].values

#initialize weights(randomly)
w_0 = rand.uniform(-1, 1)
w = np.random.uniform(-1, 1, x.shape[1]) #random weights for each feature
#Learning rate
alpha = 0.01
num_iterations = 5000

#run gradient descent
print("Training logistic regression model...")
w, w_0, predictions_raw = gradient_descent(x, y, w, w_0, alpha, num_iterations)

# Make predictions
#returns bools
def predict(x, w, w_0):
    z = np.dot(x, w) + w_0
    return sigmoid(z) >= 0.5

# Test predictions only binary values
# test_x = np.array([1, 1, 0, 1])
# predictions = predict(test_x, w, w_0)

# # Output results
# print("Test prediction: \n")
# for i, pred in enumerate(predictions):
#     result = "Admitted" if pred else "Not Admitted"
#     print(f"{test_x[i]}: {result}")

# Predictions on the dataset
predictions = predict(x, w, w_0)

# Evaluate model
m = len(y)
predictions = predict(x, w, w_0)
# print(predictions)
print("Dataset predictions: \n")
for i, pred in enumerate(predictions):
    result = "Toxic" if pred else "Non-Toxic"
    print(f"{x[i]}: {result}")

#if prediction is 1 and y is 1, it is a true positive
#if prediction is 0 and y is 0, it is a true negative
#if prediction is 1 and y is 0, it is a false positive
#if prediction is 0 and y is 1, it is a false negative
true_negatives = 0
true_positives = 0
false_negatives = 0
false_positives = 0

for i in range(m):
    if predictions[i] == 1 and y[i] == 1:
        true_positives += 1
    elif predictions[i] == 0 and y[i] == 0:
        true_negatives += 1
    elif predictions[i] == 1 and y[i] == 0:
        false_positives += 1
    elif predictions[i] == 0 and y[i] == 1:
        false_negatives += 1

# Accuracy
accuracy = (true_positives + true_negatives) / m
print("Accuracy:", accuracy)

# Compute precision, recall, and F1-score
#if sigmoid value is greater than 0.5, it is considered as 1
#need to check for division by zero
if true_positives == 0:
    precision = 0
    recall = 0
    f1_score = 0
else:
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Compute confusion matrix
cm = confusion_matrix(y, predictions)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Toxic', 'Toxic'])
disp.plot(cmap='coolwarm')
plt.title('Confusion Matrix')
plt.show()