# SparkDataFrames

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [License](#license)

## Introduction
This project demonstrates the use of Spark DataFrames for various data manipulation and machine learning tasks. It includes:
- Setting up SparkContext and SparkSession.
- Loading and preprocessing data.
- Building and evaluating machine learning models using Spark MLlib.

## Installation
To install the required dependencies, use the following commands:
```sh
pip install pyspark
pip install findspark
pip install pandas
```

## Usage
1. Clone the repository.
2. Install the required dependencies as mentioned above.
3. Run the `SparkDataframes.ipynb` notebook to execute the code.

## Project Structure
- `SparkDataframes.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, and evaluation.

## Data Loading and Preprocessing
The project starts with loading and preprocessing a CSV dataset using Pandas and Spark.

### Code Example:
```python
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext
import findspark

findspark.init()

# Load data using Pandas
df = pd.read_csv("cleaned_music_streaming.csv")
df.head()
```

### Create Spark Context and Session:
```python
# Context
SparkContext = SparkContext()

# Session
spark = SparkSession \
    .builder \
    .appName("Python Spark DataFrames basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)
spark_df.printSchema()
spark_df.show()
```

## Model Training
The project includes training several machine learning models using Spark MLlib, such as k-Nearest Neighbors, Decision Tree, Random Forest, and Logistic Regression.

### Example:
```python
from pyspark.ml.classification import KNNClassifier, DecisionTreeClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.feature import VectorAssembler

# VectorAssembler to combine features
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol='features')
data = assembler.transform(spark_df)

# Split data into training and testing sets
train_df, test_df = data.randomSplit([0.8, 0.2])

# Define models
knn = KNNClassifier(k=3, featuresCol='features', labelCol='genre')
dt = DecisionTreeClassifier(featuresCol='features', labelCol='genre')
rf = RandomForestClassifier(featuresCol='features', labelCol='genre')
lr = LogisticRegression(featuresCol='features', labelCol='genre')

# Fit models
knnModel = knn.fit(train_df)
dtModel = dt.fit(train_df)
rfModel = rf.fit(train_df)
lrModel = lr.fit(train_df)
```

## Evaluation
The models are evaluated using accuracy metrics.

### Code Example:
```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Make predictions
predictions_knn = knnModel.transform(test_df)
predictions_dt = dtModel.transform(test_df)
predictions_rf = rfModel.transform(test_df)
predictions_lr = lrModel.transform(test_df)

# Evaluate accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="genre", predictionCol="prediction", metricName="accuracy")
accuracy_knn = evaluator.evaluate(predictions_knn)
accuracy_dt = evaluator.evaluate(predictions_dt)
accuracy_rf = evaluator.evaluate(predictions_rf)
accuracy_lr = evaluator.evaluate(predictions_lr)

print("k-Nearest Neighbors Accuracy:", accuracy_knn)
print("Decision Tree Accuracy:", accuracy_dt)
print("Random Forest Accuracy:", accuracy_rf)
print("Logistic Regression Accuracy:", accuracy_lr)
```

## License
This project was given and managed by German International Univeristy
