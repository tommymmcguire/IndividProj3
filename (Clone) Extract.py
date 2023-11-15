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
