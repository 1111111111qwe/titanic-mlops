import mlflow
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

tracking_uri = "file://" + os.path.abspath("mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("titanic")

# Load data
df = pd.read_csv("data/titanic.csv")

# Keep only useful columns
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)

# Convert Sex column from text to numbers
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])

# Split into input and output
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accuracy", acc)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Accuracy: {acc}")
    print(f"Run ID: {run.info.run_id}")