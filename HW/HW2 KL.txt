# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 18:28:42 2020

@author: tsjlk
"""

# In[1]:

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
spark = SparkSession.builder.appName('k-means clustering').getOrCreate()
# Loads data
dataset = spark.read.csv("hack_data.csv",header=True,inferSchema=True)
dataset = dataset.select([c for c in dataset.columns if c != "Location"])
dataset.head()
dataset.describe().show()



# In[2]:
# ## Format the Data
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols = dataset.columns, outputCol='features')
feature_data = vec_assembler.transform(dataset)

# In[3]:
# ## Scale the Data
# It is a good idea to scale our data to deal with the curse of dimensionality: https://en.wikipedia.org/wiki/Curse_of_dimensionality
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(feature_data)
# Normalize each feature to have unit standard deviation.
final_data = scalerModel.transform(feature_data)


# In[4]:
# ## Train the Model and Evaluate
from pyspark.ml.clustering import KMeans
# Trains a k-means model.
kmeans2 = KMeans(featuresCol='scaledFeatures', k=2)
model2 = kmeans2.fit(final_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse2 = model2.computeCost(final_data)

# Trains a k-means model.
kmeans3 = KMeans(featuresCol='scaledFeatures', k=3)
model3 = kmeans3.fit(final_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse3 = model3.computeCost(final_data)

# Trains a k-means model.
kmeans4 = KMeans(featuresCol='scaledFeatures', k=4)
model4 = kmeans4.fit(final_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse4 = model4.computeCost(final_data)

# Trains a k-means model.
kmeans5 = KMeans(featuresCol='scaledFeatures', k=5)
model5 = kmeans5.fit(final_data)

# Evaluate clustering by computing Within Set Sum of Squared Errors.
wssse5 = model5.computeCost(final_data)

# In[5]:
# Result
print("Within Set Sum of Squared Errors where K = 2 is " + str(wssse2))
print("Within Set Sum of Squared Errors where K = 3 is " + str(wssse3))
print("Within Set Sum of Squared Errors where K = 4 is " + str(wssse4))
print("Within Set Sum of Squared Errors where K = 5 is " + str(wssse5))

print("\nBased on the observation, because of the elbow effect, the third hacker is highly possible to be involved in the attacks. Furthermore, also based on the observation, a fourth hacker attacker might also be involved in the attack.")