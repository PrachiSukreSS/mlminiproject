#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Music Genre Classification using KNN
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Dataset Creation
np.random.seed(42)

n_samples = 200
data = {
    'Danceability': np.random.rand(n_samples),
    'Energy': np.random.rand(n_samples),
    'Tempo': np.random.randint(60, 200, n_samples),
    'Loudness': np.random.uniform(-60, 0, n_samples),
    'Acousticness': np.random.rand(n_samples),
    'Instrumentalness': np.random.rand(n_samples),
    'Valence': np.random.rand(n_samples),  # mood positivity
    # length of song
    'Duration_ms': np.random.randint(150000, 300000, n_samples),
    'Genre': np.random.choice(['Pop', 'Rock', 'Classical', 'Jazz'], n_samples)
}

df = pd.DataFrame(data)
print("Sample Data:\n", df.head(), "\n")

# Data Preprocessing
X = df.drop('Genre', axis=1)
y = df['Genre']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training & Evaluation


def evaluate_knn(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    return acc, preds


# Test different k values
k_values = range(1, 15)
accuracies = []
for k in k_values:
    acc, _ = evaluate_knn(k)
    accuracies.append(acc)

best_k = k_values[np.argmax(accuracies)]
print(f"Best k value: {best_k} with Accuracy = {max(accuracies)*100:.2f}%")

# Train final model with best k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Classification Report
print("\n Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.title("Confusion Matrix - Music Genre Classification")
plt.xlabel("Predicted Genre")
plt.ylabel("Actual Genre")
plt.show()

# Accuracy vs K Visualization
plt.figure(figsize=(7, 4))
plt.plot(k_values, accuracies, marker='o', color='purple')
plt.title("K Value vs Model Accuracy")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Feature Correlation Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop('Genre', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 2D Visualization of Clusters (Danceability vs Energy)
plt.figure(figsize=(7, 5))
sns.scatterplot(x='Danceability', y='Energy',
                hue='Genre', data=df, palette='Spectral')
plt.title("Music Genre Distribution")
plt.show()


# In[ ]:
