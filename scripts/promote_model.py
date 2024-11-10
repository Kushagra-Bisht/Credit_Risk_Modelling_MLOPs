# promote model
import os
import mlflow

def promote_model():
    # Set up DagsHub token
    dagshub_token = "10a7c2f7d30e69f138e738ca411a1fbd78583d48"
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # Set MLflow tracking URI
    mlflow.set_tracking_uri('https://dagshub.com/Kushagra-Bisht/Credit_Risk_Modelling_MLOPs.mlflow')

    # Initialize Mlflow client
    client = mlflow.MlflowClient()
    model_name = "my_model"

    try:
        # Get the latest version in staging
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        
        if not staging_versions:
            print("No model in 'Staging' stage found.")
            return

        latest_version_staging = staging_versions[0].version

        # Archive the current production model
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )

        # Promote the new model to production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
        )
        print(f"Model version {latest_version_staging} promoted to Production")

    except mlflow.exceptions.RestException as e:
        print(f"Error accessing model versions: {e}")

if __name__ == "__main__":
    promote_model()
