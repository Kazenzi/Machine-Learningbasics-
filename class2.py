import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the CSV file
df = pd.read_csv('E:\kcauni\Year 4\sem 2\Machine learning\Machine learning basics\Irislec.csv') 

# Feature scaling
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Separate features
x = df.loc[:, features].values

# Separate out the target
y = df.loc[:, ["Species"]].values

# Standardize features
x = StandardScaler().fit_transform(x)

# PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principalComponent1', 'principalComponent2'])

# Explained Variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Sepal Length vs Species
sns.scatterplot(x=principalDf['principalComponent1'], y=principalDf['principalComponent2'], hue=df['Species'], ax=axes[0, 0])
axes[0, 0].set_title('PCA Components vs Species')

# Sepal Width vs Species
sns.scatterplot(x='SepalWidthCm', y='Species', data=df, ax=axes[0, 1], hue='Species')
axes[0, 1].set_title('Sepal Width vs Species')

# Petal Length vs Species
sns.scatterplot(x='PetalLengthCm', y='Species', data=df, ax=axes[1, 0], hue='Species')
axes[1, 0].set_title('Petal Length vs Species')

# Petal Width vs Species
sns.scatterplot(x='PetalWidthCm', y='Species', data=df, ax=axes[1, 1], hue='Species')
axes[1, 1].set_title('Petal Width vs Species')

plt.tight_layout()
plt.show()