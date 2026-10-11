# =====================================================
# STEP 1: IMPORT LIBRARIES AND LOAD DATASET
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# Load Kaggle Iris dataset
df = pd.read_csv(BASE_DIR / "iris.csv")

# Display first 5 rows
print("\nIris Dataset:")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nDataset Columns:")
print(df.columns)

# =====================================================
# STEP 2: DATA INSPECTION & PREPROCESSING
# =====================================================

# Check dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Check statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check unique species
print("\nUnique Species:")
print(df['Species'].unique())


# =====================================================
# STEP 3: SEPARATE FEATURES FOR CLUSTERING
# =====================================================

# Remove unnecessary ID column if present
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Separate flower measurements from species labels
X = df.drop('Species', axis=1)
y = df['Species']

print("\nFeatures Used for Clustering:")
print(X.head())

print("\nFeature Shape:", X.shape)

print("\nSpecies Labels:")
print(y.unique())


# =====================================================
# STEP 4: FEATURE SCALING
# =====================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Features:")
print(X_scaled[:5])

print("\nScaled Data Shape:", X_scaled.shape)


# =====================================================
# STEP 5: K-MEANS CLUSTERING
# =====================================================

from sklearn.cluster import KMeans

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train the model and assign clusters
clusters = kmeans.fit_predict(X_scaled)

# Store cluster assignments
df['Cluster'] = clusters

print("\nCluster Assignments:")
print(df[['Species', 'Cluster']].head(10))

print("\nNumber of Samples in Each Cluster:")
print(df['Cluster'].value_counts().sort_index())


# # =====================================================
# # STEP 6: VISUALIZE K-MEANS CLUSTERS USING PCA
# # =====================================================

from sklearn.decomposition import PCA

# Reduce 4 features to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap='viridis',
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering of Iris Dataset (PCA)")
plt.colorbar(label="Cluster")

plt.show()

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Explained:")
print(pca.explained_variance_ratio_.sum())

# =====================================================
# STEP 7: HIERARCHICAL CLUSTERING
# =====================================================

from sklearn.cluster import AgglomerativeClustering

# Create Hierarchical Clustering model
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

# Assign clusters
hierarchical_clusters = hierarchical.fit_predict(X_scaled)

# Store cluster assignments
df['Hierarchical_Cluster'] = hierarchical_clusters

print("\nHierarchical Cluster Assignments:")
print(df[['Species', 'Hierarchical_Cluster']].head(10))

print("\nNumber of Samples in Each Hierarchical Cluster:")
print(df['Hierarchical_Cluster'].value_counts().sort_index())

# =====================================================
# STEP 8: VISUALIZE HIERARCHICAL CLUSTERS
# =====================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=hierarchical_clusters,
    cmap='plasma',
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Hierarchical Clustering of Iris Dataset (PCA)")
plt.colorbar(label="Cluster")

plt.show()


# =====================================================
# STEP 9: CLUSTERING EVALUATION
# =====================================================

from sklearn.metrics import silhouette_score

# Evaluate K-Means
kmeans_silhouette = silhouette_score(X_scaled, clusters)

# Evaluate Hierarchical Clustering
hierarchical_silhouette = silhouette_score(
    X_scaled,
    hierarchical_clusters
)

print("\nClustering Evaluation Results:")

print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")

print(
    f"Hierarchical Clustering Silhouette Score: "
    f"{hierarchical_silhouette:.4f}"
)


# =====================================================
# STEP 10: COMPARE CLUSTERS WITH ACTUAL SPECIES
# =====================================================

print("\nK-Means Cluster vs Actual Species:")
print(pd.crosstab(df['Species'], df['Cluster']))

print("\nHierarchical Cluster vs Actual Species:")
print(pd.crosstab(df['Species'], df['Hierarchical_Cluster']))



# =====================================================
# STEP 11: FIND OPTIMAL NUMBER OF CLUSTERS
# =====================================================

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print(f"Clusters: {k}, Silhouette Score: {score:.4f}")


# =====================================================
# STEP 12: PCA EXPLAINED VARIANCE ANALYSIS
# =====================================================

from sklearn.decomposition import PCA

# Apply PCA to scaled features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Display total variance explained
total_variance = pca.explained_variance_ratio_.sum()

print("\nTotal Variance Explained:")
print(f"{total_variance:.2%}")

# Visualize explained variance
plt.figure(figsize=(7, 5))

plt.bar(
    ['PC1', 'PC2'],
    pca.explained_variance_ratio_
)

plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance")

plt.show()


# =====================================================
# STEP 13: TRAIN-TEST SPLIT
# =====================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)



# =====================================================
# STEP 14: CLASSIFICATION MODEL
# =====================================================

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nInitial Model Accuracy:")
print(model.score(X_test, y_test))



# =====================================================
# STEP 15: CROSS-VALIDATION
# =====================================================

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(cv_scores)

print("\nMean Cross-Validation Accuracy:")
print(f"{cv_scores.mean():.4f}")



# =====================================================
# STEP 16: HYPERPARAMETER TUNING
# =====================================================

from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__C": [0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Score:")
print(f"{grid_search.best_score_:.4f}")



# =====================================================
# STEP 17: CLASSIFICATION EVALUATION
# =====================================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

y_pred = best_model.predict(X_test)

print("\nFinal Test Accuracy:")
print(f"{accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=best_model.classes_
)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_model.classes_
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.show()



# =====================================================
# STEP 18: ROC-AUC EVALUATION
# =====================================================

from sklearn.metrics import roc_auc_score

y_prob = best_model.predict_proba(X_test)

roc_auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovr",
    average="weighted",
    labels=best_model.classes_
)

print("\nMulticlass ROC-AUC Score:")
print(f"{roc_auc:.4f}")



# =====================================================
# STEP 19: FINAL RESULTS SUMMARY
# =====================================================

print("\n========================================")
print("       WEEK 3 FINAL RESULTS")
print("========================================")

print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")

print(
    f"Hierarchical Silhouette Score: "
    f"{hierarchical_silhouette:.4f}"
)

print(
    f"PCA Variance Explained: "
    f"{pca.explained_variance_ratio_.sum():.2%}"
)

print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Multiclass ROC-AUC: {roc_auc:.4f}")

print("\nWeek 3 Implementation Completed!")