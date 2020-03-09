# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 00:01:30 2020

@author: Kyle
"""

# In[1]
# Create entry points to spark
from pyspark.sql import SparkSession
spark = SparkSession.builder\
           .appName("HW1 KL") \
           .config("spark.some.config.option", "some-value")\
           .config("spark.driver.bindAddress", "127.0.0.1")\
           .getOrCreate()
           
# In[2]
# Import data
           
ad = spark.read.csv('C:\Github\Big-Data-Practice\HW1\cruise_ship_info.csv', header=True, inferSchema=True)
ad.show(10)

# In[3]
# Transform data structure

from pyspark.ml.linalg import Vectors
ad_df = ad.rdd.map(lambda x: [Vectors.dense(x[2:-2]), x[-1]]).toDF(['features', 'crew'])
ad_df.show(10)

# In[4]:
# Build linear regression model

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol = 'crew')

# In[5]:
# Fit the model

lr_model = lr.fit(ad_df)

# In[6]:
# Prediction

pred = lr_model.transform(ad_df)
pred.show(10)

# In[7]:
# Module evaluation

from pyspark.ml.evaluation import RegressionEvaluator 
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='crew')
evaluator.evaluate(pred)

# In[8]:
def modelsummary(model, param_names):
    import numpy as np
    print ("Note: the last rows are the information for Intercept")
    print ("##","-------------------------------------------------")
    print ("##","  Estimate   |   Std.Error  |   t Values  |  P-value")
    coef = np.append(list(model.coefficients), model.intercept)
    Summary=model.summary
    param_names.append('intercept')

    for i in range(len(Summary.pValues)):
        print ("##",'{:10.6f}'.format(coef[i]),\
        '{:14.6f}'.format(Summary.coefficientStandardErrors[i]),\
        '{:12.3f}'.format(Summary.tValues[i]),\
        '{:12.6f}'.format(Summary.pValues[i]),\
        param_names[i])

    print ("##",'---')
    print ("##","Mean squared error: % .6f" \
           % Summary.meanSquaredError, ", RMSE: % .6f" \
           % Summary.rootMeanSquaredError )
    print ("##","Multiple R-squared: %f" % Summary.r2, "," )
    print ("##","Multiple Adjusted R-squared: %f" % Summary.r2adj, ", \
            Total iterations: %i"% Summary.totalIterations)

param_names = ad.columns[2:-2]
modelsummary(lr_model, param_names)

# In[9]:
# ## Linear regression with cross-validation
# ## Training and test datasets

training, test = ad_df.randomSplit([0.8, 0.2], seed=123)
lr_model = lr.fit(training)
pred = lr_model.transform(training)
pred.show(10)
param_names = ad.columns[2:-2]
modelsummary(lr_model, param_names)

# Make predictions.
pred_test = lr_model.transform(test)
pred_test.show(10)

from pyspark.ml.evaluation import RegressionEvaluator
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="crew",
                                predictionCol="prediction",
                                metricName="rmse")
# metricName Supports: - "rmse" (default): root mean squared error - 
# "mse": mean squared error - 
# "r2": R^2^ metric - 
# "mae": mean absolute error

rmse = evaluator.evaluate(pred_test)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# In[10]

# This result is reliable because the RMSE of the result is low


