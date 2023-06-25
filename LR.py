
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn import linear_model
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import copy
import regex as re


loan_train = pd.read_csv("loan_train.csv")
loan_test = pd.read_csv("loan_test.csv")


# Set a random seed

random_seed = 94
np.random.seed(random_seed)


# ### Implementing SMOTE for imbalanced data


x = loan_train.loc[:, loan_train.columns != 'Label']
y = loan_train.loc[:, loan_train.columns == 'Label']

x_test = loan_test.loc[:, loan_test.columns != 'Label']
y_test = loan_test.loc[:, loan_test.columns == 'Label']


sm = SMOTE(random_state = 42)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = random_seed)

columns = x_train.columns

x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)
x_train_sm = pd.DataFrame(data = x_train_sm, columns = columns)
y_train_sm = pd.DataFrame(data = y_train_sm, columns = ['Label'])

print("\033[1m Length of oversampled data is", len(x_train_sm))
print("\033[1m Number of 'Good' in oversampled data", len(y_train_sm[y_train_sm['Label'] == 0]))
print("\033[1m Number of 'Bad'", len(y_train_sm[y_train_sm['Label'] == 1]))
print("\033[1m Proportion of 'Good' data in oversampled data is", len(y_train_sm[y_train_sm['Label'] == 0])/len(x_train_sm))
print("\033[1m Proportion of 'Bad' data in oversampled data is", len(y_train_sm[y_train_sm['Label'] == 1])/len(x_train_sm))


# ### Finding the most important features in the dataset
loan_cols = loan_train.columns.values.tolist()
y = ['Label']
x = [i for i in loan_cols if i not in y]

logreg = LogisticRegression(solver = 'lbfgs', max_iter = 10)
rfe = RFE(logreg)
rfe = rfe.fit(x_train_sm, y_train_sm.values.ravel())
print(rfe.support_)
print(rfe.ranking_)


data_X1 = pd.DataFrame({
 'Feature': x_train_sm.columns,
 'Importance': rfe.ranking_},)

data_X1.sort_values(by = ['Importance'])


cols = []
for i in range (0, len(data_X1['Importance'])): 
    if data_X1['Importance'][i] == 1:
        cols.append(data_X1['Feature'][i])

print(cols)
print(len(cols))


x = x_train_sm[cols]
y = y_train_sm['Label']


# ### Implementing Logistic Regression


# Define the range of regularization parameters to explore
reg = [0.001, 0.01, 0.1, 1, 10]


# #### Fitting the train dataset


# Perform hyperparameter tuning with cross-validation
logreg_cv = LogisticRegressionCV(Cs = reg, cv = 10)

logreg_cv.fit(x_train_sm, y_train_sm.values.ravel())


# Get the best regularization parameter found during cross-validation
best_reg_param = logreg_cv.C_[0]

print("Best Regularization Parameter:", best_reg_param)


# #### Predicting the values for the validation dataset


# Predict using the best model on the validation data

y_val_pred = logreg_cv.predict(x_val)


# Evaluate model performance on the validation data

val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Calculate predicted probabilities for the bad class
y_val_prob = logreg_cv.predict_proba(x_val)[:, 1]


# Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_val, y_val_prob)


# Print the ROC AUC and confusion matrix
print("ROC AUC:", auc)




# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# Calculate the confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the confusion matrix as an image
im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)

# Add colorbar
cbar = ax.figure.colorbar(im, ax = ax)

# Set labels, title, and ticks
ax.set(xticks = np.arange(cm.shape[1]),
       yticks = np.arange(cm.shape[0]),
       xlabel = 'Predicted label',
       ylabel = 'True label',
       title = 'Confusion Matrix')

# Loop over data dimensions and create text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha = "center", va = "center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

# Display the plot
plt.show()

cm = classification_report(y_val, y_val_pred)

print(cm)


# #### Predicting for the test dataset


# Predict using the best model on the test data

y_test_pred = logreg_cv.predict(x_test)


# Evaluate model performance on the test data

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_test_pred)

# Print the ROC AUC 
print("ROC AUC:", auc)



# Plot the ROC curve
plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()



# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the confusion matrix as an image
im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)

# Add colorbar
cbar = ax.figure.colorbar(im, ax = ax)

# Set labels, title, and ticks
ax.set(xticks = np.arange(cm.shape[1]),
       yticks = np.arange(cm.shape[0]),
       xlabel = 'Predicted label',
       ylabel = 'True label',
       title = 'Confusion Matrix')

# Loop over data dimensions and create text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha = "center", va = "center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

# Display the plot
plt.show()

cm = classification_report(y_test, y_test_pred)

print(cm)


# ### Logistic Regression on Scaled Data


# #### Scaling the Data


def scaling(x, y):
    scale = preprocessing.RobustScaler()
    x_t1 = scale.fit_transform(x, y)
    x_t1 = pd.DataFrame(x_t1, columns = x.columns)
    
    scale = preprocessing.StandardScaler()
    x_t2 = scale.fit_transform(x_t1, y)
    x_t2 = pd.DataFrame(x_t2, columns = x.columns)
    
    return(x_t2)


x_train_scaled = scaling(x_train_sm, y_train_sm)
x_test_scaled = scaling(x_test, y_test)
x_val_scaled = scaling(x_val, y_val)


# #### Setting the regularisation parameter


# Define the range of regularization parameters to explore
reg_scaled = [0.001, 0.01, 0.1, 1, 10]


# #### Fitting the train data 


# Perform hyperparameter tuning with cross-validation
logreg_cv_scaled = LogisticRegressionCV(Cs = reg_scaled, cv = 10)

logreg_cv_scaled.fit(x_train_scaled, y_train_sm.values.ravel())


# Get the best regularization parameter found during cross-validation
best_reg_param = logreg_cv_scaled.C_[0]

print("Best Regularization Parameter:", best_reg_param)


# #### Predicting the values for the validation dataset


# Predict using the best model on the validation data

y_val_pred = logreg_cv_scaled.predict(x_val_scaled)


# Evaluate model performance on the validation data

val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Calculate predicted probabilities for the bad class
y_val_prob = logreg_cv_scaled.predict_proba(x_val_scaled)[:, 1]


# Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_prob)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_val, y_val_prob)

# Print the ROC AUC
print("ROC AUC:", auc)


# Plot the ROC curve
plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()



# Calculate the confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the confusion matrix as an image
im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)

# Add colorbar
cbar = ax.figure.colorbar(im, ax = ax)

# Set labels, title, and ticks
ax.set(xticks = np.arange(cm.shape[1]),
       yticks = np.arange(cm.shape[0]),
       xlabel = 'Predicted label',
       ylabel = 'True label',
       title = 'Confusion Matrix')

# Loop over data dimensions and create text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha = "center", va = "center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

# Display the plot

plt.show()

cm = classification_report(y_val, y_val_pred)

print(cm)


# #### Predicting values for test data


# Predict using the best model on the test data

y_test_pred = logreg_cv_scaled.predict(x_test_scaled)


# Evaluate model performance on the test data

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_test_pred)

# Print the ROC AUC and confusion matrix
print("ROC AUC:", auc)


# Plot the ROC curve
plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the confusion matrix as an image
im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)

# Add colorbar
cbar = ax.figure.colorbar(im, ax = ax)

# Set labels, title, and ticks
ax.set(xticks = np.arange(cm.shape[1]),
       yticks = np.arange(cm.shape[0]),
       xlabel = 'Predicted label',
       ylabel = 'True label',
       title = 'Confusion Matrix')

# Loop over data dimensions and create text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha = "center", va = "center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

# Display the plot
plt.show()

cm = classification_report(y_test, y_test_pred)

print(cm)


