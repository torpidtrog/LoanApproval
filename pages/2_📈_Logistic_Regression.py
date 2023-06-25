import streamlit as st
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


st.set_page_config(
    page_title="Model 2 Demo", 
    page_icon="📈",
    layout="wide"
)

st.markdown("# Logistic Regression")




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


correlated_features = set()
correlation_matrix = x_train_sm.corr()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

x_train_sm.drop(labels=correlated_features, axis=1,inplace=True)
x_test.drop(labels=correlated_features, axis=1,inplace=True)
x_val.drop(labels=correlated_features, axis=1,inplace=True)


# Define the range of regularization parameters to explore
reg = [0.001, 0.01, 0.1, 1, 10]


# #### Fitting the train dataset


# Perform hyperparameter tuning with cross-validation
logreg_cv = LogisticRegressionCV(Cs = reg, cv = 10)

logreg_cv.fit(x_train_sm, y_train_sm.values.ravel())


# Get the best regularization parameter found during cross-validation
best_reg_param = logreg_cv.C_[0]

print("Best Regularization Parameter:", best_reg_param)


#### Predicting the values for the validation dataset


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


option1 =st.selectbox("Unscaled or Scaled Data:", options=['Unscaled data', 'Scaled data'])


if option1 == 'Unscaled data':
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


    fig,ax = plt.subplots(
                figsize=(10,5)
                )
    # Plot the ROC curve
    plt.plot(fpr, tpr, label = 'ROC Curve (AUC = %.2f)'% auc)
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    fig.savefig("ROC1.png")
    st.image("ROC1.png")


    #######################################
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Create a figure and axis
    fig, ax = plt.subplots(
        figsize=(5,5)

    )
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
    fig.savefig("CM1.png")
    st.image("CM1.png")


    classification_report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_dict)
    st.subheader("Classification Report")
    st.table(classification_report_df)

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


if option1 == 'Scaled data':
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

    fig, ax = plt.subplots(
        figsize=(10,5)
    )
    # Plot the ROC curve
    plt.plot(fpr, tpr, label = 'ROC Curve (AUC = %.2f)'% auc)
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    fig.savefig("ROC2.png")
    st.image("ROC2.png")



    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Create a figure and axis
    fig, ax = plt.subplots(
        figsize=(5,5)

    )

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
    fig.savefig("CM2.png")
    st.image("CM2.png")


    classification_report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    classification_report_df = pd.DataFrame(classification_report_dict)
    st.subheader("Classification Report")
    st.table(classification_report_df)


