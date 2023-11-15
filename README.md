# Individual Project 3

## Using Databricks to perform ETL (Extract, Transform, Load)
---
## Overview
**Walk Through Youtube Video**
[YouTube](https://youtu.be/0_62e7htGqQ)
---
This repository contains the cloned version of my code hosted in Azure Databricks. You will need an Azure account to run this. The goal of this project is to use Databricks to extract, transform, and load data, use Delta Lake, use Spark SQL, create a visualization, and automate a trigger to initiate the pipeline. My specific goal took this assigment a step further and aimed to predict 'outage_count' for storms using MAPE as the evaluation metric. MAPE is specifically difficult to use as a metric becuase the true values of outage count are often 0 which drives MAPE to infinity. However, I am treating the project as though I would be presenting the project to someomne with a non-technical background, and therefore, MAPE would be the appropriate metric to convey the results. This will help utility companies assess the danger of storms, and allocate the resources necessary to handle them.

## Project Specifics
---
The EDA directory contains trial code. It is a bit messy. This was not included in the Pipeline, and should not be used. 
1. Extract - takes data from a parquet file and extracts the data. It then performs two simple tasks which are removing duplicate entries in 2019 and removing all rows with SimStartDate after 2019-01-01 and event_type == 'thunderstorm'.
2. Transform - calls the Extract code to run, splits the data into a train set and test set, takes a portion of the train set and creates a Spark SQL database (due to the size of the data), and creates a visualization of the data.
3. Load - Performs further data manipulation and creates 2 csv files (features.csv and outage_count.csv). Sorts and groups events by the storm start date. Takes selected features that are most important to predicting outage_count. Trains a Random forrest regression model, fits it to the data, and returns the MAPE.


Keep in mind that this is the first attempt at this project, so MAPE is fairly high. I plan to further perform EDA (exploratory data analysis) and model training/selection to reduce MAPE. 


### Visualization of average outage counts per storm type

<img width="668" alt="Screenshot 2023-11-15 at 2 09 21 PM" src="https://github.com/tommymmcguire/IndividProj3/assets/141086024/ae68a2c1-df88-4647-bf47-4e4b7b888113">

### Data Pipeline

<img width="801" alt="Screenshot 2023-11-15 at 4 36 08 PM" src="https://github.com/tommymmcguire/IndividProj3/assets/141086024/c64f33b0-dad7-4c0d-8fea-37aa7cf0434d">

### Success

<img width="808" alt="Screenshot 2023-11-15 at 5 03 38 PM" src="https://github.com/tommymmcguire/IndividProj3/assets/141086024/113e6911-7530-4d9d-a2c5-4ee923707438">


### Automating the trigger using by creating a schedule

<img width="879" alt="Screenshot 2023-11-15 at 5 04 05 PM" src="https://github.com/tommymmcguire/IndividProj3/assets/141086024/e574e029-2595-44a7-9304-e4c25468fee4">


### Success

<img width="1213" alt="Screenshot 2023-11-15 at 5 53 27 PM" src="https://github.com/tommymmcguire/IndividProj3/assets/141086024/14c7b234-3d06-4d9f-b65f-b703ae47a7f9">





