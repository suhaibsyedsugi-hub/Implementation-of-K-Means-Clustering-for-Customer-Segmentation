# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
*/
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: S SYED SUHAIB
RegisterNumber:  25013757
*/

#Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset (You can replace this with your own dataset)
# Example: Mall_Customers.csv
df = pd.read_csv(r"C:\Users\acer\Downloads\Mall_Customers.csv")

# Step 3: Basic data exploration
print(df.head())
print(df.info())

#Drop non-numeric or irrelevant columns
df = df.drop(['CustomerID', 'Gender'], axis=1)

# Step 4: Data preprocessing (scaling)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Step 5: Elbow Method to determine optimal K
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Step 6: Apply K-Means with chosen K
optimal_k = 5  # Assume 5 from the elbow method
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Step 7: Add cluster labels to original data
df['Cluster'] = clusters

# Step 8: Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2',
    s=100
)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1–100)')
plt.legend()
plt.show()

```

## Output:
<img width="713" height="411" alt="image" src="https://github.com/user-attachments/assets/c3881151-9b0c-446b-a0f0-989b08ca1b3f" />




<img width="951" height="580" alt="image" src="https://github.com/user-attachments/assets/fa08e697-e3bf-4134-95ea-fd0542fbcbfe" />




<img width="980" height="674" alt="image" src="https://github.com/user-attachments/assets/b9340a4b-aee4-4613-b19c-ed4bf148f69b" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
