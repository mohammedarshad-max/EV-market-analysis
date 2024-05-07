#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd
import numpy as np


# In[116]:


ev_features_df = pd.read_csv('ElectricCarData_Norm.csv')


# In[117]:


ev_features_df.head()


# In[118]:


# Display basic information about the dataset
ev_features_df.info()



# In[119]:


# Check for missing values
ev_features_df.isnull().sum()


# In[120]:


missing_values = ev_features_df.isnull().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])


# In[121]:


duplicate_rows = ev_features_df.duplicated()
print("Number of duplicate rows:", duplicate_rows.sum())


# # Encoding the features

# In[122]:


# Extract numerical component from "Range" feature
ev_features_df['Range_numerical'] = ev_features_df['Range'].str.extract(r'(\d+\.?\d*)').astype(float)

# Drop the original "Range" column
ev_features_df.drop(columns=['Range'], inplace=True)

# Display the updated DataFrame
ev_features_df.head()


# In[123]:


from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Encode the "Brand" feature
ev_features_df['Brand_encoded'] = label_encoder.fit_transform(ev_features_df['Brand'])

# Display the encoded DataFrame
print(ev_features_df[['Brand', 'Brand_encoded']].head())


# In[124]:


from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Convert the "Model" feature to string data type
ev_features_df['Model'] = ev_features_df['Model'].astype(str)

# Encode the "Model" feature
ev_features_df['Model_encoded'] = label_encoder.fit_transform(ev_features_df['Model'])

# Display the mapping between original values and encoded labels
print("Mapping between Original Values and Encoded Labels:")
for original_value, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{original_value}: {encoded_label}")


# In[125]:


from sklearn.preprocessing import LabelEncoder

# Extract numerical values from "TopSpeed" feature
ev_features_df['TopSpeed_numerical'] = ev_features_df['TopSpeed'].str.extract('(\d+)').astype(float)
# Drop the original "Range" column
ev_features_df.drop(columns=['TopSpeed'], inplace=True)

# Display the updated DataFrame
ev_features_df.head()





# In[126]:


# Convert "Accel" feature in DataFrame
ev_features_df['Accel'] = ev_features_df['Accel'].str.split().str[0].astype(float)

# Display the DataFrame to verify the conversion
print(ev_features_df['Accel'].head())


# In[127]:


# Convert "Efficiency" feature in DataFrame
ev_features_df['Efficiency'] = ev_features_df['Efficiency'].str.split().str[0].astype(float)

# Display the DataFrame to verify the conversion
print(ev_features_df['Efficiency'].head())


# In[128]:


import numpy as np

# Replace "-" with NaN
ev_features_df['FastCharge'] = ev_features_df['FastCharge'].replace('-', np.nan)

# Convert "FastCharge" feature in DataFrame to float
ev_features_df['FastCharge'] = ev_features_df['FastCharge'].str.split().str[0].astype(float)

# Display the DataFrame to verify the conversion
print(ev_features_df['FastCharge'].head())


# In[129]:


# Replace values in "RapidCharge" column with binary indicators
ev_features_df['RapidCharge'] = ev_features_df['RapidCharge'].map({'Rapid charging possible': 1, 'Rapid charging not possible': 0})

# Display the DataFrame to verify the modification
print(ev_features_df['RapidCharge'].head())



# In[130]:


# Map values to numerical labels
powertrain_mapping = {'All Wheel Drive': 0, 'Front Wheel Drive': 1, 'Rear Wheel Drive': 2}

# Replace values in "PowerTrain" column with numerical labels
ev_features_df['PowerTrain'] = ev_features_df['PowerTrain'].map(powertrain_mapping)

# Display the DataFrame to verify the modification
print(ev_features_df['PowerTrain'].head())


# In[131]:


import pandas as pd

# Handle NaN values in the "PlugType" column (fill NaN with a placeholder value)
ev_features_df['PlugType'].fillna('Unknown', inplace=True)

# Perform one-hot encoding on the "BodyStyle" feature
ev_features_df = pd.get_dummies(ev_features_df, columns=['BodyStyle'], prefix='BodyStyle')

# Display the updated DataFrame
print(ev_features_df.head())



# In[132]:


# Perform one-hot encoding on the "PlugType" feature
ev_features_df = pd.get_dummies(ev_features_df, columns=['PlugType'], prefix='PlugType')

# Display the updated DataFrame
print(ev_features_df.head())


# In[133]:


# Perform one-hot encoding on the "Segment" feature
ev_features_df = pd.get_dummies(ev_features_df, columns=['Segment'], prefix='Segment')

# Display the updated DataFrame
print(ev_features_df.head())


# In[134]:


ev_features_df.head()


# In[135]:


# Select the columns representing the segments
segment_columns = ['Segment_A', 'Segment_B', 'Segment_C', 'Segment_D', 'Segment_E', 'Segment_F', 'Segment_N',"Segment_S"]  # Adjust as per your DataFrame

# Visualize feature distributions within each segment
for column in segment_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=ev_features_df, x=column, y='Brand_encoded')  # Replace 'Feature_of_Interest' with the feature you want to analyze
    plt.title(f'Distribution of Feature_of_Interest across {column}')
    plt.xlabel(column)
    plt.ylabel('Feature_of_Interest')
    plt.show()


# In[136]:


# Define the segment columns
segment_columns = [col for col in ev_features_df.columns if 'Segment' in col]

# Define the numerical features for profiling
numerical_features = ['Accel', 'Efficiency', 'FastCharge', 'Seats', 'PriceEuro']

# Profile each segment
for segment in segment_columns:
    print(f"Segment: {segment}")
    segment_data = ev_features_df[ev_features_df[segment] == 1]
    segment_profile = segment_data[numerical_features].describe()
    print(segment_profile)
    print("\n")


# In[137]:


from sklearn.cluster import KMeans

import warnings

# Suppress FutureWarning and UserWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# Select features for clustering
X = ev_features_df[['Accel','Efficiency','PriceEuro']]

# Define the number of clusters
num_clusters = 5

# Initialize KMeans model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit KMeans to the data
kmeans.fit(X)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_




# In[138]:


import matplotlib.pyplot as plt
import seaborn as sns

# Add cluster labels to the DataFrame
ev_features_df['Cluster'] = cluster_labels

# Visualize clusters in 2D feature space (Accel vs. Efficiency)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=ev_features_df, x='Accel', y='Efficiency', hue='Cluster', palette='viridis', s=100)
plt.title('Clusters in Feature Space (Accel vs. Efficiency)')
plt.xlabel('Acceleration')
plt.ylabel('Efficiency')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Analyze cluster characteristics
cluster_centers = kmeans.cluster_centers_
cluster_characteristics = pd.DataFrame(cluster_centers, columns=['Accel', 'Efficiency', 'PriceEuro'])
cluster_characteristics['Cluster'] = range(1, num_clusters+1)

# Display cluster characteristics
print("Cluster Characteristics:")
print(cluster_characteristics)


# In[139]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualize clusters in 3D space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of data points colored by cluster labels
ax.scatter(X['Accel'], X['Efficiency'], X['PriceEuro'], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)

# Plot centroids of clusters
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidths=5, color='r')

ax.set_xlabel('Acceleration')
ax.set_ylabel('Efficiency')
ax.set_zlabel('Price (Euro)')

plt.title('Clusters in 3D Feature Space')
plt.show()



# In[140]:


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize Cluster Distribution
plt.figure(figsize=(8, 6))
sns.scatterplot(data=X, x='Accel', y='Efficiency', hue=cluster_labels, palette='viridis', legend='full')
plt.title('Cluster Distribution')
plt.xlabel('Acceleration')
plt.ylabel('Efficiency')
plt.show()

# Visualize Cluster Profiles
cluster_data = pd.concat([X, pd.Series(cluster_labels, name='Cluster')], axis=1)
plt.figure(figsize=(10, 8))
sns.pairplot(data=cluster_data, hue='Cluster', palette='viridis')
plt.suptitle('Cluster Profiles')
plt.show()

# Analyze Cluster Centroids
cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Accel', 'Efficiency', 'PriceEuro'])
cluster_centroids['Cluster'] = range(1, num_clusters+1)
print("Cluster Centroids:")
print(cluster_centroids)

# Generate Insights
print("Insights:")
for i in range(num_clusters):
    cluster_size = (cluster_labels == i).sum()
    print(f"Cluster {i+1}: Size={cluster_size}")

# Validate Clustering Solution (Optional)
from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {silhouette_avg}")


# In[141]:


# Retrieve the cluster centroids
cluster_centroids = kmeans.cluster_centers_

# Create a DataFrame to store the centroid values
centroid_df = pd.DataFrame(cluster_centroids, columns=X.columns)

# Add a column for cluster labels
centroid_df['Cluster'] = range(1, num_clusters + 1)

# Display the centroid values
print("Cluster Centroids:")
print(centroid_df)

# Visualize the centroid values
plt.figure(figsize=(10, 6))
for feature in X.columns:
    plt.plot(centroid_df['Cluster'], centroid_df[feature], marker='o', label=feature)
plt.title("Cluster Centroids")
plt.xlabel("Cluster")
plt.ylabel("Feature Value")
plt.legend()
plt.grid(True)
plt.show()


# In[142]:


import pandas as pd

# Calculate cluster centroids
cluster_centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)

# Add cluster labels to the centroids
cluster_centroids['Cluster'] = range(1, num_clusters + 1)

# Merge cluster labels with original data
ev_features_df['Cluster'] = cluster_labels + 1  # Adding 1 to labels to match cluster numbers

# Iterate over each cluster
for cluster_num in range(1, num_clusters + 1):
    print(f"Cluster {cluster_num}:")
    cluster_data = ev_features_df[ev_features_df['Cluster'] == cluster_num]
    
    # Calculate descriptive statistics for each feature
    cluster_profile = cluster_data.describe()
    print(cluster_profile)
    
    # Visualize feature distributions
    for column in cluster_data.columns:
        if column != 'Cluster':
            plt.figure(figsize=(8, 6))
            sns.histplot(data=cluster_data, x=column, kde=True)
            plt.title(f'Distribution of {column} in Cluster {cluster_num}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()


# In[ ]:




