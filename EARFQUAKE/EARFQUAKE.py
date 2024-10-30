from streamlit import st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import os as os

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.semi_supervised import LabelPropagation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

df = pd.read_csv("mount/src/earfquake/earthquakes.csv")
df.head()

df

file_path = 'mount/src/earfquake/earthquakes.csv'
data = pd.read_csv(file_path)

data_info = {
    "head": data.head(),
    "info": data.info(),
    "description": data.describe(),
    "null_values": data.isnull().sum()
}

data_info

#MAGNITUDE DISTRIBUTION
plt.figure(figsize=(10, 6))
sns.kdeplot(data['magnitude'], shade=True)
plt.title("Magnitude Distribution (KDE)")
plt.xlabel("Magnitude")
plt.ylabel("Density")
plt.show()
st.pyplot(plt)


#MAGNITUDE VS DEPTH
plt.figure(figsize=(10, 6))
plt.hexbin(data['magnitude'], data['depth'], gridsize=30, cmap="YlGnBu")
plt.colorbar(label="Frequency")
plt.title("Magnitude vs Depth Hexbin Plot")
plt.xlabel("Magnitude")
plt.ylabel("Depth (km)")
plt.show()
st.pyplot(plt)


#DEPTH DISTRIBUTION
plt.figure(figsize=(10, 6))
sns.violinplot(y=data['depth'])
plt.title("Depth Distribution (Violin Plot)")
plt.ylabel("Depth (km)")
plt.show()
st.pyplot(plt)


#EARTHQUAKE LOCATIONS
plt.figure(figsize=(12, 8))
sns.kdeplot(x=data['longitude'], y=data['latitude'], cmap="Reds", shade=True, thresh=0.05)
plt.title("Earthquake Location Density Plot")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
st.pyplot(plt)


#EARTHQUAKE MAGNITUDE BY CONTINENT
plt.figure(figsize=(12, 6))
sns.swarmplot(x='continent', y='magnitude', data=data)
plt.title("Magnitude Distribution by Continent")
plt.xlabel("Continent")
plt.ylabel("Magnitude")
plt.show()
st.pyplot(plt)


#MAGNITUDE BY TSUNAMI PRESENCE
plt.figure(figsize=(10, 6))
sns.histplot(data, x='magnitude', hue='tsunami', multiple="stack", bins=30)
plt.title("Magnitude Distribution by Tsunami Presence")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.show()
st.pyplot(plt)


#HOUR OF EARTHQUAKE OCCURENCE
data['time'] = pd.to_datetime(data['time'], errors='coerce')


data['hour'] = data['time'].dt.hour


hours = data['hour'].value_counts().sort_index()
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
theta = np.linspace(0, 2 * np.pi, len(hours))
ax.bar(theta, hours, width=0.3)
ax.set_xticks(theta)
ax.set_xticklabels(hours.index)
plt.title("Earthquake Occurrences by Hour (Polar Plot)")
plt.show()
st.pyplot(plt)


#MAGNITUDE VS DEPTH BY TYPE 
sns.pairplot(data, vars=['magnitude', 'depth'], hue='type', palette="husl")
plt.suptitle("Magnitude and Depth by Earthquake Type", y=1.02)
plt.show()
st.pyplot(plt)


#MAGNITUDE DISTRIBUTION BY DISTANCE FROM EPICENTER
plt.figure(figsize=(10, 6))
sns.regplot(x='distanceKM', y='magnitude', data=data, scatter_kws={'alpha':0.5})
plt.title("Magnitude vs. Distance from Epicenter")
plt.xlabel("Distance from Epicenter (KM)")
plt.ylabel("Magnitude")
plt.show()
st.pyplot(plt)
