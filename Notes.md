# Table of Contents
- [01 - Get Started with Notebooks](#01---get-started-with-notebooks)
- [02 - Get AutoML Prediction](#02---get-autoML-prediction)
- [03 - Get Designer Prediction](#03---get-designer-prediction)
- [04 - Run Experiments](#04---run-experiments)
- [05 - Train Models](#05---train-models)


# 01 - Get Started with Notebooks

## Easiest way to connect to a workspace
```python
from azureml.core import Workspace

ws = Workspace.from_config()
```

# 02 - Get AutoML Prediction

## Use `requests` to get predictions from an AutoML endpoint
```python
import json
import requests

#Features for a patient
x = [{"PatientID": 1,
      "Pregnancies": 5,
      "PlasmaGlucose": 181.0,
      "DiastolicBloodPressure": 90.6,
      "TricepsThickness": 34.0,
      "SerumInsulin": 23.0,
      "BMI": 43.51,
      "DiabetesPedigree": 1.21,
      "Age": 21.0}]

#Create a "data" JSON object
input_json = json.dumps({"data": x})

#Set the content type and authentication for the request
headers = {"Content-Type":"application/json",
           "Authorization":"Bearer " + key}

#Send the request
response = requests.post(endpoint, input_json, headers=headers)
```

# 03 - Get Designer Prediction
## Use `urllib` to get predictions from a Designer Pipeline
```python
import urllib.request
import json
import os

data = {
    "Inputs": {
        "WebServiceInput0":
        [
            {
                    'PatientID': 1882185,
                    'Pregnancies': 9,
                    'PlasmaGlucose': 104,
                    'DiastolicBloodPressure': 51,
                    'TricepsThickness': 7,
                    'SerumInsulin': 24,
                    'BMI': 27.36983156,
                    'DiabetesPedigree': 1.3504720469999998,
                    'Age': 43,
            },
        ],
    },
    "GlobalParameters":  {
    }
}

body = str.encode(json.dumps(data))


headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ key)}

req = urllib.request.Request(endpoint, body, headers)
```

# 04 - Run Experiments
## Start logging in an experiment run
```python
from azureml.core import Experiment

experiment = Experiment(workspace=ws, name="mslearn-diabetes")
run = experiment.start_logging()
```

## Four logging methods
```python
run.log('observations', row_count)

run.log_image(name='label distribution', plot=fig)

run.log_list('pregnancy categories', pregnancies)

run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')
```

## Create an environment
```python
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create a Python environment for the experiment
env = Environment("experiment_env")

# Ensure the required packages are installed (we need pip and Azure ML defaults)
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults'])
env.python.conda_dependencies = packages
```

## Create a training script
```python
from azureml.core import Experiment, ScriptRunConfig

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='diabetes_experiment.py',
                                environment=env) 

# submit the experiment
experiment = Experiment(workspace=ws, name='mslearn-diabetes')
run = experiment.submit(config=script_config)
```

## Formula for tracking with MLflow
```python
import mlflow

# Set the MLflow tracking URI to the workspace
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

mlflow.set_experiment(experiment.name)

# start the MLflow experiment
with mlflow.start_run():
    mlflow.log_metric('observations', row_count)
```

# 05 - Train Models
## Create a parameterized training script
```python
import argparse

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg
```
## Run the training script with arguments
```python
from azureml.core import Experiment, ScriptRunConfig

# Create a script config
script_config = ScriptRunConfig(source_directory=training_folder,
                                script='diabetes_training.py',
                                arguments = ['--reg_rate', 0.1],
                                environment=sklearn_env) 

# submit the experiment
experiment_name = 'mslearn-train-diabetes'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
```


