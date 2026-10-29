import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Churn_Modelling.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Identify the target column
possible_targets = [
    "Salary", "EstimatedSalary", "Estimated Salary",
    "salary", "estimated_salary"
]
target = next((c for c in possible_targets if c in df.columns), None)

if target is None:
    raise ValueError(
        "No salary target column found. Check the dataset columns printed above."
    )

# Remove identifiers and rows with missing targets
drop_cols = [
    c for c in ["RowNumber", "CustomerId", "Surname"]
    if c in df.columns
]
df = df.drop(columns=drop_cols)
df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = pd.to_numeric(df[target], errors="coerce")
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]

# Preprocess numeric and categorical columns
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2:", r2_score(y_test, predictions))

# Save trained model and feature information
joblib.dump(model, os.path.join(ARTIFACT_DIR, "salary_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(ARTIFACT_DIR, "feature_columns.pkl"))

print("Training complete. Model saved in:", ARTIFACT_DIR)