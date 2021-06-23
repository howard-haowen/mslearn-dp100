
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
- **Quantiles**: The quantile method assigns values to bins based on percentile ranks. This method is also known as equal height binning.
- **Equal Width**: With this option, you must specify the total number of bins. The values from the data column are placed in the bins such that each bin has the same interval between starting and ending values. As a result, some bins might have more values if data is clumped around a certain point.
- **Custom Edges**: You can specify the values that begin each bin. *The edge value is always the lower boundary of the bin*.
![binning](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwingpage.net%2Fcontent%2FExam%2FCOMPS382%2520Exam.html&psig=AOvVaw2T0TRM-xEJCAnlqv-dB66G&ust=1624547284247000&source=images&cd=vfe&ved=0CAoQjRxqFwoTCOCYtruErvECFQAAAAAdAAAAABAD)

2. 


### Normalize Data 
Partition and Sample 
Remove Duplicate Rows 
SMOTE 
Select Columns Transform 
Select Columns in Dataset 
Split Data
