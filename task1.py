import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#Load the dataset
titanic=pd.read_csv("Titanic-Dataset.csv")
#EDA
print("first 5 rows are:",titanic.head())
print('===============================================================================')
print("info of the data:")
titanic.info()
print('===============================================================================')
print("description of the data:\n",titanic.describe())
print('===============================================================================')
print("value count=\n",titanic['Sex'].value_counts())
print('===============================================================================')
#Drop uneccesary colunms
titanic.drop(['PassengerId','Name','Ticket'], axis=1, inplace=True)
#Count Plot
sns.countplot(x='Sex',data=titanic)
plt.title('Sex Distribution')
plt.show()
#PiePlot
titanic['Survived'].value_counts().plot.pie(autopct="%1.1f%%")
plt.title("Survival Distribution")
plt.show()
#histograms
fig,axis=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
fig.set_size_inches(10,10)
sns.histplot(titanic['Age'],kde=True,ax=axis[0][0])
sns.histplot(titanic['Fare'],kde=True,ax=axis[0][1])
sns.histplot(titanic['SibSp'],kde=True,ax=axis[1][0])
sns.histplot(titanic['Parch'],kde=True,ax=axis[1][1])
plt.show()
#Scatterplots
sns.set_style('darkgrid')
plt.title('Age vs Fare')
sns.scatterplot(data=titanic,x='Age',y='Fare',hue='Survived')
plt.show()
#Box Plot
sns.boxplot(x='Pclass',y='Age',data=titanic)
plt.show()
#Violin Plot
sns.violinplot(x='Survived',y='Fare',data=titanic)
plt.show()
print('===============================================================================')
#Handling Missing Values
titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean())
titanic['Cabin']=titanic['Cabin'].fillna(titanic['Cabin'].mode()[0])
titanic['Embarked']=titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
#Data Preparation
X=titanic[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]
y=titanic['Survived']
#Convert categorical data into numerical data
X = pd.get_dummies(X, drop_first=True)
#Train_Test_Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train=\n",X_train.shape)
print("X_test=\n",X_test.shape)
print('===============================================================================')
print("y_train=\n",y_train.shape)
print("y_test=\n",y_test.shape)
print('===============================================================================')
#Chech the missing values
print("Missing values in training set;\n",X_train.isnull().sum())
#Model Training
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
print('Model trained succesfully')
print('===============================================================================')
#Prediction
y_pred=model.predict(X_test)
print("Predicted Values-\n",y_pred)
print('===============================================================================')
#Model Evaluation
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy=",accuracy)
print('===============================================================================')
print("Classification Report=\n",classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix=\n",cm)
print('===============================================================================')
#Visualization of output
sns.heatmap(cm,annot=True,cmap='Blues',fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print('===============================================================================')
#Feature Importance
importances=model.feature_importances_
feature_names=X.columns
feature_importance_df=pd.DataFrame({'Feature':feature_names,'Importance':importances})
feature_importance_df=feature_importance_df.sort_values(by='Importance',ascending=False)
print('Feature Importances:\n',feature_importance_df)
#Plot Feature Importance
sns.barplot(x='Importance',y='Feature',data=feature_importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
print('===============================================================================')