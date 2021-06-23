
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
- **Set of thresholds**
  - **ClipPeaks**: When you clip values by peaks, you specify only an upper boundary. Values greater than that boundary value are replaced. >>> *Lower threshold*
  - **ClipSubpeaks**: When you clip values by subpeaks, you specify only a lower boundary. Values that are less than that boundary value are replaced.
  - **ClipPeaksAndSubpeaks**: When you clip values by peaks and subpeaks, you can specify both the upper and lower boundaries. Values that are outside that range are replaced. Values that match the boundary values are not changed.

### Edit Metadata 
### Group Data into Bins 
### Normalize Data 
Partition and Sample 
Remove Duplicate Rows 
SMOTE 
Select Columns Transform 
Select Columns in Dataset 
Split Data
