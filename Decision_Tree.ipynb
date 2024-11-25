# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

# Step 1: Load the dataset
# Assuming you have the dataset in a CSV file, adjust the file path accordingly
# Social_Network_Ads.csv is assumed to have columns: 'User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'

dataset = pd.read_csv('Social_Network_Ads.csv')

# Step 2: Data Preprocessing
# Dropping 'User ID' since it is not relevant to our analysis
X = dataset[['Age', 'EstimatedSalary']].values
y = dataset['Purchased'].values

# Step 3: Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4: Feature Scaling (optional for Decision Tree but good practice)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Training the Decision Tree Classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Step 6: Predicting the Test set results
y_pred = classifier.predict(X_test)

# Step 7: Evaluating the Model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print("Confusion Matrix:\n", cm)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Step 8: Visualizing the Decision Tree
plt.figure(figsize=(12,8))
tree.plot_tree(classifier, filled=True, feature_names=['Age', 'EstimatedSalary'], class_names=['Not Purchased', 'Purchased'])
plt.title('Decision Tree Visualization')
plt.show()
