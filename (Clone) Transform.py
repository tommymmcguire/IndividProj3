# Databricks notebook source
# MAGIC %run "./Extract"

# COMMAND ----------

train_data_df = df[df['SimStartDate'] < '2018-11-01']
test_data_df = df[df['SimStartDate'] >= '2018-11-01']

# COMMAND ----------

train_data_sample = train_data_df.sample(frac=0.5, random_state=42)

# COMMAND ----------

# Check if the directory exists and remove it if it does
dir_path = "dbfs:/user/hive/warehouse/train_data_sample"
if any(item.path == dir_path for item in dbutils.fs.ls("dbfs:/user/hive/warehouse/")):
    dbutils.fs.rm(dir_path, True)
# Create the directory for the table
dbutils.fs.mkdirs(dir_path)
# Create a table from the CSV file using Spark SQL with 'IF NOT EXISTS'
spark.sql("""
  CREATE TABLE IF NOT EXISTS train_data_sample
  USING CSV
  OPTIONS (
    path '/user/hive/warehouse/train_data_sample/train_data_sample.csv',
    header 'true',
    inferSchema 'true',
    delimiter ','
  )
""")


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df is your DataFrame and you want to compare the mean of 'feature1' across different 'categories'
sns.barplot(x='outage_count', y='event_type', data=df)
plt.xlabel('Outage_count')
plt.ylabel('Average of event_type')
plt.title('Average of outage_count for each event_type')
plt.show()

