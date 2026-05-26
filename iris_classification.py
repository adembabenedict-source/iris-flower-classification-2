# ============================================
# PROFESSIONAL IRIS FLOWER CLASSIFICATION
# WITH MULTIPLE MODELS AND ADVANCED GRAPHS
# ============================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# LOAD DATASET
# -----------------------------
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

print(df.head())

# -----------------------------
# DATA VISUALIZATION
# -----------------------------

# 1. Pairplot
sns.pairplot(df, hue='species', palette='Set1')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# -----------------------------
# 2. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    linewidths=1
)

plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# 3. Boxplots
# -----------------------------
plt.figure(figsize=(12, 6))

for i, column in enumerate(feature_names):
    plt.subplot(2, 2, i + 1)

    sns.boxplot(
        x='species',
        y=column,
        data=df,
        palette='Set2'
    )

    plt.title(f"Boxplot of {column}")

plt.tight_layout()
plt.show()

# -----------------------------
# 4. Histograms
# -----------------------------
df.hist(figsize=(12, 8), color='skyblue')
plt.suptitle("Feature Distributions")
plt.show()

# -----------------------------
# 5. Violin Plots
# -----------------------------
plt.figure(figsize=(12, 8))

for i, column in enumerate(feature_names):
    plt.subplot(2, 2, i + 1)

    sns.violinplot(
        x='species',
        y=column,
        data=df,
        palette='Pastel1'
    )

    plt.title(f"Violin Plot of {column}")

plt.tight_layout()
plt.show()

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# FEATURE SCALING
# -----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# MODEL DEFINITIONS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

# -----------------------------
# TRAIN MODELS
# -----------------------------
accuracy_results = {}

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracy_results[name] = accuracy

    print(f"\n{name}")
    print("-" * 30)

    print("Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# -----------------------------
# 6. MODEL COMPARISON GRAPH
# -----------------------------
plt.figure(figsize=(10, 6))

sns.barplot(
    x=list(accuracy_results.keys()),
    y=list(accuracy_results.values()),
    palette='viridis'
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")

plt.ylim(0.8, 1.0)

for i, value in enumerate(accuracy_results.values()):
    plt.text(i, value + 0.005, f"{value:.2f}", ha='center')

plt.show()

# -----------------------------
# BEST MODEL: RANDOM FOREST
# -----------------------------
print("\n==============================")
print("HYPERPARAMETER TUNING")
print("==============================")

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6, None]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    params,
    cv=5
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross Validation Score:", grid.best_score_)

best_model = grid.best_estimator_

# -----------------------------
# FINAL PREDICTIONS
# -----------------------------
y_pred = best_model.predict(X_test)

# -----------------------------
# 7. CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 8. FEATURE IMPORTANCE
# -----------------------------
importance = best_model.feature_importances_

plt.figure(figsize=(8, 5))

sns.barplot(
    x=importance,
    y=feature_names,
    palette='magma'
)

plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# -----------------------------
# 9. PCA VISUALIZATION
# -----------------------------
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame()

pca_df['PCA1'] = X_pca[:, 0]
pca_df['PCA2'] = X_pca[:, 1]
pca_df['species'] = y

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='species',
    palette='Set1',
    data=pca_df,
    s=100
)

plt.title("PCA Visualization of Iris Dataset")
plt.show()

# -----------------------------
# 10. CROSS VALIDATION SCORES
# -----------------------------
cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5
)

print("\n==============================")
print("CROSS VALIDATION")
print("==============================")

print("Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# -----------------------------
# 11. CLASS DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6, 4))

sns.countplot(
    x='species',
    data=df,
    palette='Set2'
)

plt.title("Class Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 12. RADAR CHART
# -----------------------------
mean_values = df.groupby('species').mean()

labels = feature_names

angles = np.linspace(
    0,
    2 * np.pi,
    len(labels),
    endpoint=False
)

angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

for i in range(len(mean_values)):

    values = mean_values.iloc[i].tolist()
    values += values[:1]

    ax.plot(angles, values, label=target_names[i])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title("Radar Chart of Iris Features")
plt.legend()
plt.show()

# -----------------------------
# FINAL MESSAGE
# -----------------------------
print("\n========================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("Professional Iris Classification Built")
print("========================================")