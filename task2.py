import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print ("First 5 rows:")
print (df.head());
sns.scatterplot(x=df['petal length (cm)'], y=df['petal width (cm)'], hue=iris.target_names[y])
plt.title ("Petal Length vs Petal Width")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print ("Training samples:", X_train.shape)

print ("Testing samples:", X_test.shape)

print ("================================================")
knn = KNeighborsClassifier (n_neighbors=5)

knn.fit (X_train, y_train)
print ("KNN Model trained successfully")
print ("================================================")
y_pred = knn.predict(X_test)
accuracy = accuracy_score (y_test, y_pred)
print ("Classification Report:")
print ("Accuracy:", accuracy)
print ("================================================")

print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix:")
print(cm)
print ("================================================")
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)

plt.xlabel ("Predicted")
plt.ylabel ("Actual")
plt.title ("Confusion Matrix - KNN")
plt.show()