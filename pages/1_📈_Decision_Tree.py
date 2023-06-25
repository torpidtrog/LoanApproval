import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Decision Tree", 
    page_icon="📈",
    layout="wide"
)

st.markdown("# Decision Tree")
train = pd.read_csv("loan_train.csv")
test = pd.read_csv("loan_test.csv")

x_train = train.drop(['Label'],axis=1)
y_train = train['Label']
x_test = test.drop(['Label'],axis=1)
y_test = test['Label']

correlated_features = set()
correlation_matrix = x_train.corr()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)


x_train.drop(labels=correlated_features, axis=1, inplace=True)
x_test.drop(labels=correlated_features, axis=1, inplace=True)


clf = DecisionTreeRegressor(random_state=44)
model = clf.fit(x_train, y_train)
y_test_pred = clf.predict(x_test)

# Calculate AUC score on the validation set
val_auc = metrics.roc_auc_score(y_test, y_test_pred)
print("Validation AUC:", val_auc)


# Evaluate model performance on the test data

test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)

# Classification Report
classification_report_dict = metrics.classification_report(y_test, y_test_pred, output_dict=True)
cm =metrics.confusion_matrix(y_test, y_test_pred)

# Calculate the false positive rate, true positive rate, and thresholds for the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

# Calculate the area under the ROC curve
auc = roc_auc_score(y_test, y_test_pred)

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
fig.savefig("ROC3.png")
st.image("ROC3.png")


# Create a figure and axis
fig, ax = plt.subplots(
        figsize=(5,5)
    )
im = ax.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
cbar = ax.figure.colorbar(im, ax = ax)
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
fig.savefig("CM3.png")
st.image("CM3.png")

# Classification Report
cr = metrics.classification_report(y_test, y_test_pred)
classification_report_df = pd.DataFrame(classification_report_dict)
st.subheader("Classification Report")
st.table(classification_report_df)