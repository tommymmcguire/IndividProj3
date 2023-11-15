# Databricks notebook source
# Run this before any other code cell
# This downloads the data files into the same directory where you have saved this notebook

import urllib.request
from pathlib import Path
import os
path = Path()

# Dictionary of file names and download links
files = {'outage_data.parquet':'https://storage.googleapis.com/aipi_datasets/outage_data.parquet'}

# Download each file
for key,value in files.items():
    filename = path/key
    url = value
    # If the file does not already exist in the directory, download it
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url,filename)

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# Read the data into a Pandas dataframe
df = pd.read_parquet(path='./outage_data.parquet', engine='pyarrow')

# Remove duplicate entries in 2019
# Remove all rows with SimStartDate after 2019-01-01 and event_type == 'thunderstorm'
df = df.loc[~((df['SimStartDate'] > '2019-01-01') & (df['event_type'] == 'thunderstorm'))]
df.describe()

# COMMAND ----------

df.head()

# COMMAND ----------

# Save feature names to a csv file
df.columns.to_frame().to_csv('features.csv', index=False)

# COMMAND ----------

# Count the number of grid cells and outage events
# 488 grid cells and 154 outage events
df.loc[:,['lat','lon']].groupby(['lat','lon']).size().reset_index(name='count').sort_values(by='count', ascending=False)

# COMMAND ----------

# Sort events by number of outages
grouped = df.loc[:,['SimStartDate','event_type','outage_count']].groupby(['SimStartDate','event_type']).outage_count.sum().sort_values(ascending=False)
grouped = grouped.reset_index(name='outage_count')
grouped = grouped.set_index('SimStartDate')
grouped

# COMMAND ----------

# Sort events by SimStartDate
grouped = df.loc[:,['SimStartDate','event_type','outage_count']].groupby(['SimStartDate','event_type']).outage_count.sum().sort_index()
grouped = grouped.reset_index(name='outage_count')
# save to csv
grouped.to_csv('outage_count.csv')

# COMMAND ----------

selected_features = ['outage_count', 'length_proxy_30', 'fuse_counts', 'transformer_counts', 'line_length_30', 'line_length_60', 'recloser_counts', 'switch_counts', 'length_proxy_60', 'pole_counts', 'breaker_counts', 'lon', 'LU_24_30', 'LU_24_60', 'grid_id', 'LU_23_30', 'canopy_interval_11_20_30', 'LU_41_60', 'LU_21_60', 'LU_41_30', 'canopy_interval_11_20_60', 'canopy_interval_31_40_30', 'y', 'LU_23_60', 'LU_71_60', 'canopy_var_30', 'canopy_interval_1_10_30', 'LU_22_30', 'canopy_interval_41_50_30', 'LU_43_60', 'canopy_interval_51_60_30', 'canopy_interval_0_0_30', 'canopy_interval_61_70_60', 'canopy_interval_1_10_60', 'LU_31_60', 'canopy_interval_51_60_60', 'canopy_interval_41_50_60', 'canopy_interval_31_40_60', 'canopy_interval_0_0_60', 'canopy_mean_30', 'canopy_interval_21_30_60', 'LU_21_30', 'canopy_mean_60', 'canopy_interval_71_80_60', 'canopy_75th_60', 'LU_95_30', 'canopy_interval_21_30_30', 'LU_22_60', 'x', 'canopy_var_60', 'LU_95_60', 'canopy_interval_61_70_30', 'canopy_interval_71_80_30', 'LU_52_60', 'LU_42_60', 'LU_43_30', 'LU_42_30', 'canopy_75th_30']
# Group by 'SimStartDate' and 'event_type', and calculate the sum for each selected feature
gp = df.groupby(['SimStartDate', 'event_type'])[selected_features].sum()

# Reset the index to make 'SimStartDate' and 'event_type' columns again
gp = gp.reset_index()

# COMMAND ----------

import pandas as pd


# Convert 'SimStartDate' to datetime
gp['SimStartDate'] = pd.to_datetime(gp['SimStartDate'])

# Extract year, month, and day from 'SimStartDate'
gp['year'] = gp['SimStartDate'].dt.year
gp['month'] = gp['SimStartDate'].dt.month
gp['day'] = gp['SimStartDate'].dt.day

# Encode 'event_type' if it's categorical
if gp['event_type'].dtype == 'object':
    gp['event_type'] = gp['event_type'].astype('category').cat.codes


# COMMAND ----------

# Split the data
train_data = gp[gp['SimStartDate'] < '2018-11-01']
test_data = gp[gp['SimStartDate'] >= '2018-11-01']

# Separate features and target
X_train = train_data.drop(['SimStartDate', 'outage_count'], axis=1)
y_train = train_data['outage_count']
X_test = test_data.drop(['SimStartDate', 'outage_count'], axis=1)
y_test = test_data['outage_count']


# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

# Define and train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)


# COMMAND ----------

import numpy as np

# Predict on the test set
y_pred = model.predict(X_test)

# Define MAPE function
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# Calculate MAPE
mape_score = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape_score}%")


# COMMAND ----------

print(df.info())

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Check for missing values
missing_values = df.isnull().sum()

any_missing = df.isnull().values.any()
print("Are there any missing values?", any_missing)

# # Print missing values for each column
# print(missing_values)

# COMMAND ----------

# from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.preprocessing import StandardScaler

# COMMAND ----------

train_data = df[df['SimStartDate'] < '2018-11-01']
test_data = df[df['SimStartDate'] >= '2018-11-01']

# COMMAND ----------

train_data.describe()

# COMMAND ----------

# Check for missing values
missing_values = train_data.isnull().sum()

any_missing = train_data.isnull().values.any()
print("Are there any missing values?", any_missing)

# # Print missing values for each column
# print(missing_values)

# COMMAND ----------

# train_data_sample = train_data.sample(frac=0.5, random_state=42)

# COMMAND ----------

# dbutils.fs.rm("dbfs:/user/hive/warehouse/train_data_sample", True)  # Remove the table if it exists

# dbutils.fs.mkdirs("dbfs:/user/hive/warehouse/train_data_sample")  # Create the directory for the table

# # Save the pandas DataFrame to a CSV file
# train_data_sample.to_csv("/dbfs/user/hive/warehouse/train_data_sample/train_data_sample.csv", index=False)

# # Create a table from the CSV file
# spark.sql("""
#   CREATE TABLE train_data_sample
#   USING CSV
#   OPTIONS (
#     'path' '/user/hive/warehouse/train_data_sample/train_data_sample.csv',
#     'header' 'true',
#     'inferSchema' 'true',
#     'delimiter' ','
#   )
# """)

# COMMAND ----------

selected_features = ['length_proxy_30', 'fuse_counts', 'transformer_counts', 'line_length_30', 'line_length_60', 'recloser_counts', 'switch_counts', 'length_proxy_60', 'pole_counts', 'breaker_counts', 'lon', 'LU_24_30', 'LU_24_60', 'grid_id', 'LU_23_30', 'canopy_interval_11_20_30', 'LU_41_60', 'LU_21_60', 'LU_41_30', 'canopy_interval_11_20_60', 'canopy_interval_31_40_30', 'y', 'LU_23_60', 'LU_71_60', 'canopy_var_30', 'canopy_interval_1_10_30', 'LU_22_30', 'canopy_interval_41_50_30', 'LU_43_60', 'canopy_interval_51_60_30', 'canopy_interval_0_0_30', 'canopy_interval_61_70_60', 'canopy_interval_1_10_60', 'LU_31_60', 'canopy_interval_51_60_60', 'canopy_interval_41_50_60', 'canopy_interval_31_40_60', 'canopy_interval_0_0_60', 'canopy_mean_30', 'canopy_interval_21_30_60', 'LU_21_30', 'canopy_mean_60', 'canopy_interval_71_80_60', 'canopy_75th_60', 'LU_95_30', 'canopy_interval_21_30_30', 'LU_22_60', 'x', 'canopy_var_60', 'LU_95_60', 'canopy_interval_61_70_30', 'canopy_interval_71_80_30', 'LU_52_60', 'LU_42_60', 'LU_43_30', 'LU_42_30', 'canopy_75th_30']

# COMMAND ----------

# Splitting the train and test data into features and target
X_train = train_data[selected_features]
y_train = train_data['outage_count']

X_test = test_data[selected_features]
y_test = test_data['outage_count']

# COMMAND ----------

print(X_train.dtypes)
X_train.info()

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Preprocess the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test_scaled)
mape = mean_absolute_percentage_error(y_test, y_pred)


# COMMAND ----------

print(mape)

# COMMAND ----------

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_pred - y_true) / denominator
    diff[denominator == 0] = 0  # handle the case where y_true and y_pred are both zero
    return 100 * np.mean(diff)


y_pred = model.predict(X_test_scaled)
smape_score = smape(y_test, y_pred)
print(f"sMAPE: {smape_score}%")


# COMMAND ----------

print(df['outage_count'].nunique())

# COMMAND ----------

def count_unique_values(df, column_name):
    if column_name in df.columns:
        value_counts = df[column_name].value_counts()
        print(f"Unique values and their counts in '{column_name}':\n{value_counts}")
    else:
        print(f"The column '{column_name}' does not exist in the DataFrame.")


count_unique_values(df, 'outage_count')

