# Table of Contents
- [01 - Get Started with Notebooks](#01---get-started-with-notebooks)
- [02 - Get AutoML Prediction](#02---get-autoML-prediction)
- [03 - Get Designer Prediction](#03---get-designer-prediction)
- [04 - Run Experiments](#04---run-experiments)
- [05 - Train Models](#05---train-models)
- [06 - Work with Data](#06---work-with-data)
- [07 - Work with Compute](#07---work-with-compute)
- [08 - Create a Pipeline](#08---create-a-pipeline)
- [09 - Create a Real-time Inferencing Service](#09---create-a-real-time-inferencing-service)
 
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
## Two ways to register a trained model
-- With a `run` object
```python
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Parameterized script'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})
```

-- With the `Model` class
```python
from azureml.core import Model

Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'diabetes_model',
               tags={'Training context':'Pipeline'},
               properties={'AUC': np.float(auc), 'Accuracy': np.float(acc)})
```

# 06 - Work with Data
## Create a tabular dataset
```python
from azureml.core import Dataset

# Get the default datastore
default_ds = ws.get_default_datastore()

#Create a tabular dataset from the path on the datastore (this may take a short while)
tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

# Display the first 20 rows as a Pandas dataframe
tab_data_set.take(20).to_pandas_dataframe()
```

## Train a model from a tabular dataset
- In the training script
```python
parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')

# `input_datasets['training_data']` comes from `diabetes_ds.as_named_input('training_data')]`
diabetes = run.input_datasets['training_data'].to_pandas_dataframe()

# Another way to create the `diabetes` dataframe object, which does not hard-code the dataset name into the script
dataset = Dataset.get_by_id(ws, id=args.training_dataset_id)
diabetes = dataset.to_pandas_dataframe()
```
- Pass the dataset as an argument to the training script
```python
# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                              script='diabetes_training.py',
                              arguments = ['--regularization', 0.1, # Regularizaton rate parameter
                                           '--input-data', diabetes_ds.as_named_input('training_data')], # Reference to dataset
                              environment=sklearn_env) 
```

## Create a file dataset
```python
#Create a file dataset from the path on the datastore (this may take a short while)
file_data_set = Dataset.File.from_files(path=(default_ds, 'diabetes-data/*.csv'))

# Get the files in the dataset
for file_path in file_data_set.to_path():
    print(file_path)
```

## Train a model from a file dataset
- In the training script
```python
parser.add_argument('--input-data', type=str, dest='dataset_folder', help='data mount point')

# Get the training data path from the input
data_path = run.input_datasets['training_files'] 

# Read the files
all_files = glob.glob(data_path + "/*.csv")
diabetes = pd.concat((pd.read_csv(f) for f in all_files), sort=False)

# Another way to create the `data_path` path object, which does not hard-code the dataset name into the script
data_path = args.dataset_folder
```

- Pass the dataset as an argument to the training script
```python
# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='diabetes_training.py',
                                arguments = ['--regularization', 0.1, # Regularizaton rate parameter
                                             '--input-data', diabetes_ds.as_named_input('training_files').as_download()], # Reference to dataset location
                                environment=sklearn_env) # Use the environment created previously
```
- Using `as_download` causes the files in the file dataset to be downloaded to a temporary location on the compute where the script is being run, while `as_mount` creates a mount point from which the files can be streamed directly from the datasetore.

# 07 - Work with Compute
## Create a compute cluster
```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "your-compute-cluster"

try:
    # Check for existing compute target
    training_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        training_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
```

- Assign the created compute cluster to a script config
```python
# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                                script='diabetes_training.py',
                                arguments = ['--input-data', diabetes_ds.as_named_input('training_data')],
                                environment=registered_env,
                                compute_target=cluster_name) 
```

# 08 - Create a Pipeline
## Create a script for Step 1
- Input data for Step 1
```python
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')

diabetes = run.input_datasets['raw_data'].to_pandas_dataframe()
```
- Output data for Step 1
```python
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')

save_folder = args.prepped_data

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
diabetes.to_csv(save_path, index=False, header=True)
```

## Create a script for Step 2
```python
parser.add_argument("--training-data", type=str, dest='training_data', help='training data')
args = parser.parse_args()
training_data = args.training_data
```

## Create two `PythonScriptStep` instances for Step 1 and 2
```python
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create an OutputFileDatasetConfig (temporary Data Reference) for data passed from step 1 to step 2
prepped_data = OutputFileDatasetConfig("prepped_data")

# Step 1, Run the data prep script
prep_step = PythonScriptStep(name = "Prepare Data",
                                source_directory = experiment_folder,
                                script_name = "prep_diabetes.py",
                                arguments = ['--input-data', diabetes_ds.as_named_input('raw_data'),
                                             '--prepped-data', prepped_data],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

# Step 2, run the training script
train_step = PythonScriptStep(name = "Train and Register Model",
                                source_directory = experiment_folder,
                                script_name = "train_diabetes.py",
                                arguments = ['--training-data', prepped_data.as_input()],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)
```

- Use `OutputFileDatasetConfig()` for the output of Step 1, which is passed along to Step 2 by `prepped_data.as_input()`

## Create the pipeline connecting Step 1 and 2
```python
from azureml.pipeline.core import Pipeline

# Construct the pipeline
pipeline_steps = [prep_step, train_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name = 'mslearn-diabetes-pipeline')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
```

## Publish the pipeline and get its endpoint
```python
published_pipeline = pipeline_run.publish_pipeline(
    name="diabetes-training-pipeline", description="Trains diabetes model", version="1.0")
    
rest_endpoint = published_pipeline.endpoint
```

## Call the pipeline endpoint
### Get authentication
```python
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
```
### Make a post request to the run_id
```python
import requests

experiment_name = 'mslearn-diabetes-pipeline'
rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": experiment_name})
run_id = response.json()["Id"]
```

## Start a pipeline run with a run_id
```python
from azureml.pipeline.core.run import PipelineRun

published_pipeline_run = PipelineRun(ws.experiments[experiment_name], run_id)
published_pipeline_run.wait_for_completion(show_output=True)
```

## Schedule a pipeline run
```python
from azureml.pipeline.core import ScheduleRecurrence, Schedule

# Submit the Pipeline every Monday at 00:00 UTC
recurrence = ScheduleRecurrence(frequency="Week", interval=1, week_days=["Monday"], time_of_day="00:00")
weekly_schedule = Schedule.create(ws, name="weekly-diabetes-training", 
                                  description="Based on time",
                                  pipeline_id=published_pipeline.id, 
                                  experiment_name='mslearn-diabetes-pipeline', 
                                  recurrence=recurrence)                             
```

# 09 - Create a Real-time Inferencing Service
## Create a scoring/entry script 
```python
import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)
```

## Create an ACI web service 
```python
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = "diabetes-service"

service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
```
## Consume the web service
### With Python SDK, where the `service` object is available 
```python
import json

# This time our input is an array of two feature arrays
x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array or arrays to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Call the web service, passing the input data
predictions = service.run(input_data = input_json)

# Get the predicted classes.
predicted_classes = json.loads(predictions)
   
for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )
```

### With HTTP requests, which require getting the service endpoint in advance
- Get the service endpoint
```python
endpoint = service.scoring_uri
```
- Make a POST request
```python
import requests
import json

x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Set the content type
headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, input_json, headers = headers)
predicted_classes = json.loads(predictions.json())

for i in range(len(x_new)):
    print ("Patient {}".format(x_new[i]), predicted_classes[i] )
```

