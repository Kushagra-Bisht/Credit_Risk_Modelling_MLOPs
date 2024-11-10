import dagshub
import mlflow
dagshub.init(repo_owner='Kushagra-Bisht', repo_name='Credit_Risk_Modelling_MLOPs', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Kushagra-Bisht/Credit_Risk_Modelling_MLOPs.mlflow")
import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

  