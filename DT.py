import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


train = pd.read_csv("loan_train.csv")
test = pd.read_csv("loan_test.csv")

x_train = train.drop(['Label'],axis=1)
y_train = train['Label']
x_test = test.drop(['Label'],axis=1)
y_test = test['Label']

clf = DecisionTreeRegressor(random_state=44)
model = clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

# Calculate AUC score on the validation set
val_auc = metrics.roc_auc_score(y_test, predictions)
print("Validation AUC:", val_auc)

cm =metrics.confusion_matrix(y_test, predictions)
cr = metrics.classification_report(y_test, predictions)
print(cr)

# Create a figure and axis
fig, ax = plt.subplots()
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
plt.show()



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
predictions = clf.predict(x_test)

# Calculate AUC score on the validation set
val_auc = metrics.roc_auc_score(y_test, predictions)
print("Validation AUC:", val_auc)

cr = metrics.classification_report(y_test, predictions)
cm =metrics.confusion_matrix(y_test, predictions)
print(cr)


# Create a figure and axis
fig, ax = plt.subplots()
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
plt.show()

