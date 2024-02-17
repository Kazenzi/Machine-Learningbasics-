import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas.")

# Load the CSV file
df = pd.read_csv('.\loan_data.csv')

# printing the loaded data to see it 
print(df.head()) #checking the the rows 
print(df.info()) #checking the info
print(df.describe()) #checking  summary 

# visualize using seaborn
sns.set(style='whitegrid')
sns.catplot(x='Gender', hue='Loan_Status', data=df, kind='count')
sns.catplot(x='Married', hue='Loan_Status', data=df, kind='count')
sns.catplot(x='Dependents', hue='Loan_Status', data=df, kind='count')
sns.catplot(x='Education', hue='Loan_Status', data=df, kind='count')
sns.catplot(x='Self_Employed', hue='Loan_Status', data=df, kind='count')
sns.histplot(df['ApplicantIncome'], bins=50, kde=False, color='blue')
sns.catplot(x='Property_Area', hue='Loan_Status', data=df, kind='count')
#plt.show()

# from the above plots we can observe that:gender, married, self employed, applicants income and property area are good attributes

# now we create model
# relevant attributes
X = pd.get_dummies(df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']])
y = df['Loan_Status']

# split the data
# plit the data into 70% training and 30% testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# creating a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# display deciosn tree
# Plot the decision tree
plt.figure(figsize=(40,30))
plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()
# let's get the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.show()