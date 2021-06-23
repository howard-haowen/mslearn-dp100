
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
![zscore](data:image/png;base64,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)
- **MinMax**: The min-max normalizer linearly rescales every feature to the [0,1] interval.
- **Logistic**
- **LogNormal**
- **TanH**: All values are converted to a hyperbolic tangent.

Partition and Sample 
Remove Duplicate Rows 
SMOTE 
Select Columns Transform 
Select Columns in Dataset 
Split Data
