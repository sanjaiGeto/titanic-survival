# Step 1: Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Titanic Dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

# Step 3: Data Exploration
print("First 5 rows of the dataset:")
print(data.head())

# Basic Info about the dataset
print("\nDataset Info:")
print(data.info())

# Checking for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Step 4: Data Cleaning
# Fill missing Age values with the median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode (most common value)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column due to too many missing values
data.drop(columns=['Cabin'], inplace=True)

# Step 5: Exploratory Data Analysis (EDA)
# Visualizing survival rate
sns.countplot(x='Survived', data=data)
plt.title('Survival Rate')
plt.show()

# Distribution of Age
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# Sex vs Survival
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival by Sex')
plt.show()

# Step 6: Feature Engineering
# Convert 'Sex' to numerical values: Female = 0, Male = 1
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# Convert 'Embarked' into numerical values
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Step 7: Splitting the dataset into Training and Testing sets
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Splitting data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Building (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Model Evaluation
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

