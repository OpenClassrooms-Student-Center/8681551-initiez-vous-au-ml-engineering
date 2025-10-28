# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import numpy as np

# %%
# Create random data
np.random.seed(42)

# Generate features
n_samples = 1000
X = pd.DataFrame(
    {
        "feature1": np.random.normal(0, 1, n_samples),
        "feature2": np.random.uniform(-10, 10, n_samples),
        "feature3": np.random.choice(["A", "B", "C"], n_samples),
        "feature4": np.random.exponential(2, n_samples),
    }
)

# Generate target variable (classification)
y = np.random.randint(0, 3, n_samples)  # 3 classes: 0, 1, 2


# %%
# Split data into train and test sets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train CatBoost model
cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    random_seed=42,
    cat_features=["feature3"],
    verbose=False,
)

random_forest_model = RandomForestClassifier()


cat_model.fit(X_train, y_train)

# Make predictions
y_pred = cat_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

# Generate detailed classification report
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"F1 Score: {f1:.3f}")
print("\nClassification Report:")
print(class_report)

# %%
