The info in this note is taken straight from [Microsoft](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/module-reference)

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
