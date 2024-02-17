import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load the CSV file
df = pd.read_csv('E:\kcauni\Year 4\sem 2\Machine learning\Machine learning basics\Irislec.csv') 
iris=pd.read_csv('E:\kcauni\Year 4\sem 2\Machine learning\Machine learning basics\Irislec.csv') 

# Display the first 2 rows of the DataFrame
# print(df.head(2))
print(df)
# plot
plt.plot(iris["Id"], iris["SepalLengthCm"], "r--")  
plt.show()  # Show the plot

# scatter
iris.plot(kind="scatter", x='SepalLengthCm', y='SepalWidthCm', color='orange', label='Setosa')
iris.plot(kind="scatter", x='PetalLengthCm', y='PetalWidthCm', color='green', label='Setosa')
plt.grid()
iris.plot(kind="scatter", x='Id', y='Id', color='blue', label='Setosa')
plt.grid()

from sklearn.preprocessing import StandardScaler
# Feature scaling
features= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
# Separate features
x = df.loc[:, features].values
# Separate out the target
y= df.loc[:,["Species"]].values
# Standardize features
x = StandardScaler().fit_transform(x)
print(x, y)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principalComponent 1', 'principalComponent 2'])
print("Principal Components:")
print(principalDf.head())

# Explained Variance
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('principalComponent 1', fontsize=15)
ax.set_ylabel('principalComponent 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']

for species, color in zip(targets, colors):
    indicesToKeep = (y == species).flatten()

    ax.scatter(principalDf.loc[indicesToKeep, 'principalComponent 1'],
               principalDf.loc[indicesToKeep, 'principalComponent 2'],
               c=color, s=50, label=species)

ax.legend()
ax.grid()

plt.show()