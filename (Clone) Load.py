# Databricks notebook source
# MAGIC %run "./Extract"

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

