import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
# Load dataset
df = pd.read_csv("data/processed/airbnb_cleaned.csv")

# Extracting features for scaling and clustering
# Note: These must match exactly what was analyzed in modeling.ipynb
feature_cols = ["price", "number of reviews", "availability 365"]
X = df[feature_cols]

# Target variable for Logistic Regression
# Recreating the target as defined in modeling.ipynb (Median split)
revenue_estimate = df["price"] * df["availability 365"]
median_rev = revenue_estimate.median()
y = (revenue_estimate > median_rev).astype(int)

print("Scaling features...")
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training KMeans clusters...")
# KMeans clustering (3 clusters as identified in the notebook)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

print("Training Logistic Regression model...")
# Add cluster feature to the scaled features
X_final = np.hstack((X_scaled, clusters.reshape(-1, 1)))

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_final, y)

print("Saving model artifacts...")
# Save models
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(kmeans, "models/kmeans.joblib")
joblib.dump(model, "models/logistic_model.joblib")

print("All artifacts saved successfully in models/ directory.")
