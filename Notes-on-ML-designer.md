The info in this note is taken straight from [Microsoft Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/module-reference).

# Data preparation modules

## Data Input and Output
- move data from cloud sources into your pipeline
- write your results or intermediate data to Azure Storage, SQL Database, or Hive
- use cloud storage to exchange data between pipelines

## Data Transformation
- normalizing or binning data
- dimensionality reduction
- converting data among various file formats

### Clean Missing Data 
- Each time that you apply the Clean Missing Data module to a set of data, the same cleaning operation is applied to all columns that you select. 
- If you need to clean different columns using different methods, use separate instances of the module.
- Generate missing value indicator column: Select this option if you want to output some indication of whether the values in the column met the criteria for missing value cleaning.
- The module returns two outputs: **Cleaned dataset** and **Cleaning transformation**.

### Clip Values 
1. Select **set of thresholds**
  - **ClipPeaks**: When you clip values by peaks, you specify only an upper boundary. Values greater than that boundary value are replaced. >>> *Lower threshold*
  - **ClipSubpeaks**: When you clip values by subpeaks, you specify only a lower boundary. Values that are less than that boundary value are replaced. >>> *Upper threshold*
  - **ClipPeaksAndSubpeaks**: When you clip values by peaks and subpeaks, you can specify both the upper and lower boundaries. Values that are outside that range are replaced. Values that match the boundary values are not changed. >>> *Threshold*

2. For each threshold type, choose either of the two
- **Constant**
- **Percentile**: For example, assume you want to keep only the values in the 10-80 percentile range, and replace all others. You would choose Percentile, and then type 10 for Percentile value for lower threshold, and type 80 for Percentile value for upper threshold.

3. Define a substitute value
- **Threshold**: Replaces clipped values with the specified threshold value.
- **Mean**: Replaces clipped values with the mean of the column values. The mean is computed before values are clipped.
- **Median**: Replaces clipped values with the median of the column values. The median is computed before values are clipped.
- **Missing**: Replaces clipped values with the missing (empty) value.

### Edit Metadata 
- Treating Boolean or numeric columns as categorical values.
- Indicating which column contains the class label or contains the values you want to categorize or predict.
- Marking columns as features.
- Changing date/time values to numeric values or vice versa.
- Renaming columns.

### Group Data into Bins 
1. Select the **Binning mode**
- **Quantiles**: The quantile method assigns values to bins based on percentile ranks. This method is also known as *equal height binning*.
- **Equal Width**: With this option, you must specify the total number of bins. The values from the data column are placed in the bins such that each bin has the same interval between starting and ending values. As a result, some bins might have more values if data is clumped around a certain point.
- **Custom Edges**: You can specify the values that begin each bin. *The edge value is always the lower boundary of the bin*.
![binning](https://image.slidesharecdn.com/datapre-processing-170313100854-170315122804/95/data-pre-processing-12-638.jpg?cb=1489580903)

2. If you select the **Quantiles** binning mode, use the Quantile normalization option to determine how values are normalized before sorting into quantiles.
- **Percent**: Values are normalized within the range [0,100].
- **PQuantile**: Values are normalized within the range [0,1].
- **QuantileIndex**: Values are normalized within the range [1,number of bins].

3. If you choose the **Custom Edges** option, enter a comma-separated list of numbers to use as bin edges in the Comma-separated list of bin edges text box. For example, if you enter one bin edge value, two bins will be generated. If you enter two bin edge values, three bins will be generated.

### Join Data
- **Inner Join**: An inner join is the most common join operation. It returns the combined rows only when the values of the key columns match.
- **Left Outer Join**: A left outer join returns joined rows for all rows from the left table. When a row in the left table has no matching rows in the right table, the returned row contains missing values for all columns that come from the right table. You can also specify a replacement value for missing values.
- **Full Outer Join**: A full outer join returns all rows from the left table (table1) and from the right table (table2).
For each of the rows in either table that have no matching rows in the other, the result includes a row containing missing values.
- **Left Semi-Join**: A left semi-join returns only the values from the left table when the values of the key columns match.
![join](https://data-flair.training/blogs/wp-content/uploads/sites/2/2018/03/Types-of-Hive-joins.jpg)

### Normalize Data 
- **Zscore**

![zscore](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/aml-normalization-z-score.png)
- **MinMax**: The min-max normalizer linearly rescales every feature to the [0,1] interval.

![min-max](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/aml-normalization-minmax.png)
- **Logistic**

![logistic](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/aml-normalization-logistic.png)
- **LogNormal**

![lognormal](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/aml-normalization-lognormal.png)
- **TanH**: All values are converted to a hyperbolic tangent.

![tanh](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/aml-normalization-tanh.png)

### Partition and Sample 
- **Head**: Use this mode to get only the first n rows. 
- **Sampling**: This option supports simple random sampling or stratified random sampling.
  - **Rate of sampling**: Enter a value between 0 and 1.   
  - **Random seed for sampling**: The default value is *0, meaning that a starting seed is generated based on the system clock*. This value can lead to slightly different results each time you run the pipeline.
  - **Stratified split for sampling**: Select this option if it's important that the rows in the dataset are divided evenly by some key column before sampling.
- **Assign to Folds**: Use this option when you want to divide the dataset into subsets of the data. This option is also useful when you want to create a custom number of folds for cross-validation, or to split rows into several groups. 
- **Pick fold**: Use this option when you have divided a dataset into multiple partitions and now want to load each partition in turn for further analysis or processing. *Partition indices are 1-based*. For example, if you divided the dataset into three parts, the partitions would have the indices 1, 2, and 3.
![assign-to-folds](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/partition-and-sample.png)

### Split Data
- Use the Split Data module to divide a dataset into two distinct sets.
- This module is useful when you need to separate data into training and testing sets. 
- Splitting modes
  - Split Rows: Use this option if you just want to divide the data into two parts. You can specify the percentage of data to put in each split. By default, the data is divided 50/50.
  - Regular Expression Split: Choose this option when you want to divide your dataset by testing a single column for a value.
  ```python
   \"Text" Gryphon 
  # This example puts into the first dataset all rows that contain the text Gryphon in the column Text. It puts other rows into the second output of Split Data.
  ```
  - Relative Expression Split: Use this option whenever you want to apply a condition to a number column.
  ```python
  \"Year" > 2010
  # A common scenario is to divide a dataset by years. The following expression selects all rows where the values in the column Year are greater than 2010.
  ```
  
### SMOTE
- Synthetic Minority Oversampling Technique (SMOTE) is a statistical technique for increasing the number of cases in your dataset in a balanced way.
- The module works by generating new instances from existing **minority** cases that you supply as input. 
- This implementation of SMOTE does not change the number of **majority** cases.
- To increase the number of cases, you can set the value of **SMOTE percentage**, by using multiples of 100. For example, suppose you have an imbalanced dataset where just 1 percent of the cases have the target value A (the minority class), and 99 percent of the cases have the value B. To increase the percentage of minority cases to twice the previous percentage, you would enter 200 for SMOTE percentage.
- Use the **Number of nearest neighbors** option to determine the size of the feature space that the SMOTE algorithm uses in building new cases.
  - By increasing the number of nearest neighbors, you get features from more cases.
  - By keeping the number of nearest neighbors low, you use features that are more like those in the original sample. 

### Select Columns Transform 
- The purpose of the Select Columns Transform module is to ensure that a predictable, consistent set of columns is used in downstream machine learning operations.
- This module is helpful for tasks such as scoring, which require specific columns.
- Steps for using Select Columns Transform 
  - Add an input dataset to your pipeline in the designer.
  - Add an instance of **Filter Based Feature Selection**.
  - Connect the modules and configure the feature selection module to automatically find a number of best features in the input dataset.
  - Add an instance of **Train Model** and use the output of Filter Based Feature Selection as the input for training.
  - Attach an instance of the **Select Columns Transform module**.
  - Add the **Score Model** module.
  - Add the **Apply Transformation** module, and connect the output of the feature selection transformation.
![designer](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/filter-based-feature-selection-score.png)

## Feature Selection
### Filter Based Feature Selection
- The Filter Based Feature Selection module provides multiple feature selection algorithms to choose from. The module includes correlation methods such as Pearson correlation and chi-squared values.
- You provide a dataset and identify the column that contains the label or dependent variable. You then specify a single method to use in measuring feature importance.
- The module outputs a dataset that contains the best feature columns, as ranked by predictive power. It also outputs the names of the features and their scores from the selected metric.

#### Pearson correlation
- Requirements: Label can be text or numeric. **Features must be numeric**.
- Pearson’s correlation statistic, or Pearson’s correlation coefficient, is also known in statistical models as the r value. 
- For any two variables, it returns a value that indicates the strength of the correlation.
- Pearson's correlation coefficient is computed by *taking the covariance of two variables and dividing by the product of their standard deviations*. 
- Changes of scale in the two variables don't affect the coefficient.

![pearson](https://getcalc.com/formula/statistics/correlation-coefficient.png)

#### Chi squared
- Requirements: Labels and features can be text or numeric. Use this method for computing **feature importance for two categorical columns**.
- The two-way chi-squared test is a statistical method that measures how close expected values are to actual results. 
- The method assumes that variables are random and drawn from an adequate sample of independent variables. 
- The resulting chi-squared statistic indicates how far results are from the expected (random) result.

![chi-square](https://getcalc.com/formula/statistics/chi-squared-test.png)

### Permutation Feature Importance
- Feature values are randomly shuffled, one column at a time. 
- The performance of the model is measured before and after. You can choose one of the standard metrics to measure performance.
- The scores that the module returns represent the change in the performance of a trained model, after permutation. 
- For **Metric for measuring performance**, select a single metric to use when you're computing model quality after permutation.
  - Classification: Accuracy, Precision, Recall
  - Regression: **Precision, Recall**, Mean Absolute Error, Root Mean Squared Error, Relative Absolute Error, Relative Squared Error, Coefficient of Determination 

## Statistical Functions
### Summarize Data
- Use the Summarize Data module to create a set of standard statistical measures that describe each column in the input table.

# Machine learning algorithms
## Regression
### Boosted Decision Tree Regression
### Decision Forest Regression
### Fast Forest Quantile Regression
### Linear Regression
### Neural Network Regression
- Any class of statistical models can be termed a neural network if they use adaptive weights and can approximate non-linear functions of their inputs.
- If you accept the default neural network architecture, use the Properties pane to set parameters that control the behavior of the neural network, such as the **number of nodes in the hidden layer, learning rate, and normalization**.

### Poisson Regression
- Poisson regression is intended for predicting numeric values, typically counts.
- Counts cannot be negative. The method will fail outright if you attempt to use it with negative labels.
- A Poisson distribution is a discrete distribution; therefore, it is not meaningful to use this method with non-whole numbers.

#### Optimization tolerance
- Type a value that defines the tolerance interval during optimization. **The lower the value, the slower and more accurate the fitting**.

#### Regularization
- Regularization adds constraints to the algorithm regarding aspects of the model that are independent of the training data. Regularization is commonly used to avoid overfitting.
  - **L1 regularization** is useful if the goal is to have **a model that is as sparse as possible**. L1 regularization is done by subtracting the L1 weight of the weight vector from the loss expression that the learner is trying to minimize. The L1 norm is a good approximation to the L0 norm, which is the number of non-zero coordinates.
  - **L2 regularization** prevents any single coordinate in the weight vector from growing too much in magnitude. L2 regularization is useful if the goal is to have **a model with small overall weights**. 
  
![regularization](https://images2.programmersought.com/293/c1/c1a9b8d5574d08161922d10a5547832d.png)

## Classification
### Multiclass Boosted Decision Tree
### Multiclass Decision Forest
### Multiclass Logistic Regression
### Multiclass Neural Network
### One vs. All Multiclass
- This module creates an ensemble of binary classification models to analyze multiple classes. 
- To use this module, **you need to configure and train a binary classification model first**.

### One vs. One Multiclass
- This module is useful for creating models that predict three or more possible outcomes, when the outcome depends on continuous or categorical predictor variables.
- This module implements the one-versus-one method, in which a binary model is created per class pair. At prediction time, the class which received the most votes is selected.
- In essence, the module creates an ensemble of individual models and then merges the results, to create a single model that predicts all classes.

### Two-Class Averaged Perceptron
### Two-Class Boosted Decision Tree
### Two-Class Decision Forest
### Two-Class Logistic Regression
### Two-Class Neural Network
### Two Class Support Vector Machine
- This particular implementation is suited to prediction of two possible outcomes, based on either continuous or categorical variables.
- SVM models have been used in many applications, from information retrieval to text and image classification. 
- SVMs can be used for both classification and regression tasks.

## Clustering
### K-Means Clustering
- When you configure a clustering model by using the K-means method, you must specify a target number k that indicates the number of *centroids* you want in the model. 
- The centroid is a point that's representative of each cluster. 
- The K-means algorithm assigns each incoming data point to one of the clusters by minimizing the within-cluster *sum of squares*.

#### Number of centroids
- type the number of clusters you want the algorithm to begin with.
#### Initialization
- **First N**: Some initial number of data points are chosen from the dataset and used as the initial means. This method is also called the **Forgy method**.
- **Random**: The algorithm randomly places a data point in a cluster and then computes the initial mean to be the centroid of the cluster's randomly assigned points. This method is also called the **random partition method**.
- K-Means++: This is the default method for initializing clusters. The K-means++ algorithm was proposed in 2007 by David Arthur and Sergei Vassilvitskii to avoid poor clustering by the standard K-means algorithm. **K-means++ improves upon standard K-means by using a different method for choosing the initial cluster centers**.
#### Assign label mode
- **Ignore label column**: The values in the label column are ignored and are not used in building the model.
- **Fill missing values**: The label column values are used as features to help build the clusters. If any rows are missing a label, the value is imputed by using other features.
- **Overwrite from closest to center**: The label column values are replaced with predicted label values, using the label of the point that is closest to the current centroid.
#### Normalize features
- If you apply normalization, before training, the data points are normalized to [0,1] by MinMaxNormalizer.

# Modules for building and evaluating models
## Model Training
### Train Model
#### Classification models
- based on neural networks, decision trees, and decision forests, and other algorithms.
#### Regression models
- can include standard linear regression, or use other algorithms, including neural networks and Bayesian regression.

### Train Clustering Model
### Train PyTorch Model
1. Add **DenseNet module** or **ResNet** to your pipeline draft in the designer.
2. Add the Train PyTorch Model module to the pipeline.
3. On the left input, attach an untrained model. Attach the training dataset and validation dataset to the middle and right-hand input of Train PyTorch Model.
- For dataset, **the training dataset must be a labeled image directory**. Refer to **Convert to Image Directory** for how to get a labeled image directory.
- For **Patience**, specify how many epochs to early stop training if validation loss does not decrease consecutively. by default 3.
- For **Print frequency**, specify training log print frequency over iterations in each epoch, by default 10.
### Tune Model Hyperparameters
- In the right panel of Tune Model Hyperparameters, choose a value for **Parameter sweeping mode**. This option controls how the parameters are selected.
  - Entire grid
  - Random sweep

- When you run a parameter sweep, **the module calculates all applicable metrics for the model type** and returns them in the Sweep results report. The module uses separate metrics for regression and classification models.
- However, the metric that you choose determines how the models are ranked. **Only the top model, as ranked by the chosen metric, is output as a trained model to use for scoring**.

#### Metrics used for binary classification
- Accuracy is the proportion of true results to total cases.
- Precision is the proportion of true results to positive results.
- Recall is the fraction of all correct results over all results.
- F-score is a measure that balances precision and recall.
- AUC is a value that represents the area under the curve when false positives are plotted on the x-axis and true positives are plotted on the y-axis.
- Average Log Loss is the difference between two probability distributions: the true one, and the one in the model.

#### Metrics used for regression
- Mean absolute error averages all the errors in the model, where error means the distance of the predicted value from the true value. It's often abbreviated as MAE.
- Root of mean squared error measures the average of the squares of the errors, and then takes the root of that value. It's often abbreviated as RMSE.
- Relative absolute error represents the *error as a percentage of the true value*.
- Relative squared error normalizes the total squared error by dividing by the total squared error of the predicted values.
- **Coefficient of determination** is a single number that indicates how well data fits a model. A value of one means that the model exactly matches the data. A value of zero means that the data is random or otherwise can't be fit to the model. It's often called **r2, R2, or r-squared**.

## Model Scoring and Evaluation
### Score Model
- Use this module to generate predictions using a trained classification or regression model.
### Evaluate Model
- Use this module to measure the accuracy of a trained model. You provide a dataset containing scores generated from a model, and the Evaluate Model module computes a set of industry-standard evaluation metrics.

## Python Language
### Create Python Model module
- The module supports use of any learner that's included in the Python packages already installed in Azure Machine Learning. 
- Training code example
```python
# The script MUST define a class named AzureMLModel.
# This class MUST at least define the following three methods:
    # __init__: in which self.model must be assigned,
    # train: which trains self.model, the two input arguments must be pandas DataFrame,
    # predict: which generates prediction result, the input argument and the prediction result MUST be pandas DataFrame.
# The signatures (method names and argument names) of all these methods MUST be exactly the same as the following example.

import pandas as pd
from sklearn.naive_bayes import GaussianNB


class AzureMLModel:
    def __init__(self):
        self.model = GaussianNB()
        self.feature_column_names = list()

    def train(self, df_train, df_label):
        # self.feature_column_names records the column names used for training.
        # It is recommended to set this attribute before training so that the
        # feature columns used in predict and train methods have the same names.
        self.feature_column_names = df_train.columns.tolist()
        self.model.fit(df_train, df_label)

    def predict(self, df):
        # The feature columns used for prediction MUST have the same names as the ones for training.
        # The name of score column ("Scored Labels" in this case) MUST be different from any other columns in input data.
        return pd.DataFrame(
            {'Scored Labels': self.model.predict(df[self.feature_column_names]), 
             'probabilities': self.model.predict_proba(df[self.feature_column_names])[:, 1]}
        )

```
- Evaluation code example
```python
# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd

# The entry point function MUST have two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
    import pandas as pd
    import numpy as np

    scores = dataframe1.ix[:, ("income", "Scored Labels", "probabilities")]
    ytrue = np.array([0 if val == '<=50K' else 1 for val in scores["income"]])
    ypred = np.array([0 if val == '<=50K' else 1 for val in scores["Scored Labels"]])    
    probabilities = scores["probabilities"]

    accuracy, precision, recall, auc = \
    accuracy_score(ytrue, ypred),\
    precision_score(ytrue, ypred),\
    recall_score(ytrue, ypred),\
    roc_auc_score(ytrue, probabilities)

    metrics = pd.DataFrame();
    metrics["Metric"] = ["Accuracy", "Precision", "Recall", "AUC"];
    metrics["Value"] = [accuracy, precision, recall, auc]

    return metrics,
```
### Execute Python Script
- Azure Machine Learning uses the Anaconda distribution of Python, which includes many common utilities for data processing. We will update the Anaconda version automatically.

![flowchart](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/media/module/create-python-model.png)

## Text Analytics
### Convert Word to Vector 
- Among various word embedding technologies, in this module, we implemented three widely used methods. Two, **Word2Vec** and **FastText**, are *online-training models*. The other is a *pretrained* model, **glove-wiki-gigaword-100**.

### Extract N-Gram Features from Text

## Computer Vision
## Recommendation
## Anomaly Detection
