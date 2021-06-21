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
- [10 - Create a Batch Inferencing Service](#10---create-a-batch-inferencing-service)
- [11 - Tune Hyperparameters](#11---tune-hyperparameters)
- [12 - Use Automated Machine Learning](#12---use-automated-machine-learning)
- [13 - Explore Differential Privacy](#13---explore-differential-privacy)
- [14 - Interpret Models](#14---interpret-models) 
- [15 - Detect Unfairness](#15---detect-unfairness)

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

## Create a pipeline connecting Step 1 and 2
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

# 10 - Create a Batch Inferencing Service
## Create a scoring/entry script 
- The difference between scoring scripts for a real-time service and those for a batch inferencing service lies mostly in the `run()` function.

```python
import os
import numpy as np
from azureml.core import Model
import joblib

def init():
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)


def run(mini_batch):
    # This runs for each batch
    resultList = []

    # process each file in the batch
    for f in mini_batch:
        # Read the comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-dimensional array for prediction (model expects multiple items)
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList

```
## Create a PipelineData as the output folder for batch inferencing
```python
from azureml.pipeline.core import PipelineData

output_dir = PipelineData(name='inferences', 
                          datastore=default_ds, 
                          output_path_on_compute='diabetes/results')
```

## Create a ParallelRunStep 
```python
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep

parallel_run_config = ParallelRunConfig(
    source_directory=experiment_folder,
    entry_script="batch_diabetes.py",
    mini_batch_size="5",
    error_threshold=10,
    output_action="append_row",
    environment=batch_env,
    compute_target=inference_cluster,
    node_count=2)

parallelrun_step = ParallelRunStep(
    name='batch-score-diabetes',
    parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('diabetes_batch')],
    output=output_dir,
    arguments=[],
    allow_reuse=True
)
```
## Submit `parallelrun_step` to an experiment
```python
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline

pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
pipeline_run = Experiment(ws, 'mslearn-diabetes-batch').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)
```

## Get predictions from `pipeline_run`
```python
# Get the run for the first step and download its output
prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
```

# 11 - Tune Hyperparameters
## Sample a range of parameter values
```python
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice

params = GridParameterSampling(
    {
        # Hyperdrive will try 6 combinations, adding these as script arguments
        '--learning_rate': choice(0.01, 0.1, 1.0),
        '--n_estimators' : choice(10, 100)
    }
)
```
## Configure hyperdrive settings
```python
hyperdrive = HyperDriveConfig(run_config=script_config, 
                          hyperparameter_sampling=params, 
                          policy=None, # No early stopping policy
                          primary_metric_name='AUC', # Find the highest AUC metric
                          primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, 
                          max_total_runs=6, # Restict the experiment to 6 iterations
                          max_concurrent_runs=2) # Run up to 2 iterations in parallel
```
## Submit `hyperdrive` to an experiment
```python
# Run the experiment
experiment = Experiment(workspace=ws, name='mslearn-diabetes-hyperdrive')
run = experiment.submit(config=hyperdrive)
```

## Get the best performing run by the primary metric
```python
# Get the best run, and its metrics and arguments
best_run = run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
script_arguments = best_run.get_details() ['runDefinition']['arguments']
print('Best Run Id: ', best_run.id)
print(' -AUC:', best_run_metrics['AUC'])
print(' -Accuracy:', best_run_metrics['Accuracy'])
print(' -Arguments:',script_arguments)
```

# 12 - Use Automated Machine Learning
```python
from azureml.train.automl import AutoMLConfig

automl_config = AutoMLConfig(name='Automated ML Experiment',
                             task='classification',
                             compute_target=training_cluster,
                             training_data = train_ds,
                             validation_data = test_ds,
                             label_column_name='Diabetic',
                             iterations=4,
                             primary_metric = 'AUC_weighted',
                             max_concurrent_iterations=2,
                             featurization='auto'
                             )
```

# 13 - Explore Differential Privacy
- Epsilon: A low epsilon results in a dataset with a greater level of privacy, while a high epsilon results in a dataset that is closer to the original data. 

## Create a `smartnoise` analysis for means
```python
import opendp.smartnoise.core as sn

cols = list(diabetes.columns)
age_range = [0.0, 120.0]
samples = len(diabetes)

with sn.Analysis() as analysis:
    # load data
    data = sn.Dataset(path=data_path, column_names=cols)
    
    # Convert Age to float
    age_dt = sn.to_float(data['Age'])
    
    # get mean of age
    age_mean = sn.dp_mean(data = age_dt,
                          privacy_usage = {'epsilon': .50},
                          data_lower = age_range[0],
                          data_upper = age_range[1],
                          data_rows = samples
                         )
    
analysis.release()

# print differentially private estimate of mean age
print("Private mean age:",age_mean.value)

# print actual mean age
print("Actual mean age:",diabetes.Age.mean())
```
## Create a `smartnoise` analysis for covariance
```python
with sn.Analysis() as analysis:
    sn_data = sn.Dataset(path = data_path, column_names = cols)

    age_bp_cov_scalar = sn.dp_covariance(
                left = sn.to_float(sn_data['Age']),
                right = sn.to_float(sn_data['DiastolicBloodPressure']),
                privacy_usage = {'epsilon': 1.0},
                left_lower = 0.,
                left_upper = 120.,
                left_rows = 10000,
                right_lower = 0.,
                right_upper = 150.,
                right_rows = 10000)
analysis.release()
print('Differentially private covariance: {0}'.format(age_bp_cov_scalar.value[0][0]))
print('Actual covariance', diabetes.Age.cov(diabetes.DiastolicBloodPressure))
```
## Use SQL queries
### Create a .yml file for data schema
```python
from opendp.smartnoise.metadata import CollectionMetadata

meta = CollectionMetadata.from_file('metadata/diabetes.yml')
```

### `PandasReader` without differential privacy vs `PrivateReader` with differential privacy
```python
from opendp.smartnoise.sql import PandasReader, PrivateReader

reader = PandasReader(diabetes, meta)
private_reader = PrivateReader(reader=reader, metadata=meta, epsilon_per_column=0.7)
```

### Submit a SQL query
```python
query = 'SELECT Diabetic, AVG(Age) AS AvgAge FROM diabetes.diabetes GROUP BY Diabetic'

# Query with differential privacy
result_dp = private_reader.execute(query)
# Query without differential privacy
result = reader.execute(query)
```

# 14 - Interpret Models
## Create a SHAP model explainer 
```python
!pip show azureml-explain-model azureml-interpret

from interpret.ext.blackbox import TabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
labels = ['not-diabetic', 'diabetic']
X, y = data[features].values, data['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

model = DecisionTreeClassifier().fit(X_train, y_train)

# "features" and "classes" fields are optional
tab_explainer = TabularExplainer(model,
                             X_train, 
                             features=features, 
                             classes=labels)
```

## Get *global* feature importance
```python
# you can use the training data or the test data here
global_tab_explanation = tab_explainer.explain_global(X_train)

# Get the top features by importance
global_tab_feature_importance = global_tab_explanation.get_feature_importance_dict()
for feature, importance in global_tab_feature_importance.items():
    print(feature,":", importance)
```

## Get *local* feature importance
```python
# Get the observations we want to explain (the first two)
X_explain = X_test[0:2]

# Get predictions
predictions = model.predict(X_explain)

# Get local explanations
local_tab_explanation = tab_explainer.explain_local(X_explain)

# Get feature names and importance for each possible label
local_tab_features = local_tab_explanation.get_ranked_local_names()
local_tab_importance = local_tab_explanation.get_ranked_local_values()
```

## Upload a model explanation to the experiment output in a training script
```python
# Import libraries for model explanation
from azureml.interpret import ExplanationClient
from interpret.ext.blackbox import TabularExplainer

# Get the experiment run context
run = Run.get_context()

# Get explanation
explainer = TabularExplainer(model, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(explanation, comment='Tabular Explanation')

# Complete the run
run.complete()
```

## Download a model explanation from a `run` object
```python
from azureml.interpret import ExplanationClient

# Get the feature explanations
client = ExplanationClient.from_run(run)
engineered_explanations = client.download_model_explanation()
feature_importances = engineered_explanations.get_feature_importance_dict()

# Overall feature importance
print('Feature\tImportance')
for key, value in feature_importances.items():
    print(key, '\t', value)    
```

# 15 - Detect Unfairness
## Determine a sensitive feature (e.g. age)
```python
# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
X, y = data[features].values, data['Diabetic'].values

# Get sensitive features
S = data[['Age']].astype(int)
# Change value to represent age groups
S['Age'] = np.where(S.Age > 50, 'Over 50', '50 or younger')
# numpy.where(condition[, x, y]) returns `x` when the condition is `True` and `y` when `False`

# Split data into training set and test set
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.20, random_state=0, stratify=y)
```

## Use `MetricFrame` from `fairlearn` to calculate performance metrics by sensitive group
```python
from fairlearn.metrics import selection_rate, MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Get metrics by sensitive group from fairlearn
metrics = {'selection_rate': selection_rate, #i.e. percentage of positive predictions
           'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score}

group_metrics = MetricFrame(metrics,
                            y_test, y_hat,
                            sensitive_features=S_test['Age'])

print(group_metrics.by_group)
```

