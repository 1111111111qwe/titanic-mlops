import mlflow
import sys
import os

tracking_uri = "file://" + os.path.abspath("mlruns")
mlflow.set_tracking_uri(tracking_uri)

print(f"Tracking URI: {tracking_uri}")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()

experiment = client.get_experiment_by_name("titanic")
print(f"Experiment ID: {experiment.experiment_id}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"attributes.run_id = '{run_id}'"
)

if not runs:
    print("Run not found!")
    sys.exit(1)

accuracy = runs[0].data.metrics["accuracy"]
print(f"Accuracy: {accuracy}")

if accuracy < 0.80:
    print("FAILED: accuracy is below 0.80 threshold!")
    sys.exit(1)
else:
    print("PASSED: accuracy is above  0.80 threshold!")