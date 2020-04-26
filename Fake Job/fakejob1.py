# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:15:19 2020

@author: Anson
"""
#import os
#os.environ["JAVA_HOME"]="C:/Program Files/Java/jdk1.8.0_241"
# In[1]:data clean
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.appName('text mining').getOrCreate()


df= pd.read_csv("fake_job_postings.csv")
#df.head()
#df.describe()
#df.isna().sum()

plt.figure(figsize = (10,10))
corr = df.corr()
sns.heatmap(corr , mask=np.zeros_like(corr, dtype=np.bool) , cmap=sns.diverging_palette(-100,0,as_cmap=True) , square = True)

del df['salary_range']
del df['job_id']
df.head()

df.fillna(" ",inplace = True)
df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['department'] + ' ' + df['company_profile'] + ' ' + df['description'] + ' ' + df['requirements'] + ' ' + df['benefits'] + ' ' + df['employment_type'] + ' ' + df['required_education'] + ' ' + df['industry'] + ' ' + df['function']
df.text[0]

del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['employment_type']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']

df.head()
data=spark.createDataFrame(df)
data.show(1,truncate=False)
"""
+-------------+----------------+-------------+----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|telecommuting|has_company_logo|has_questions|fraudulent|text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
+-------------+----------------+-------------+----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|0            |1               |0            |0         |Marketing Intern US, NY, New York Marketing We're Food52, and we've created a groundbreaking and award-winning cooking site. We support, connect, and celebrate home cooks, and give them everything they need in one place.We have a top editorial, business, and engineering team. We're focused on using technology to find new and better ways to connect people around their specific food interests, and to offer them superb, highly curated information about food and cooking. We attract the most talented home cooks and contributors in the country; we also publish well-known professionals like Mario Batali, Gwyneth Paltrow, and Danny Meyer. And we have partnerships with Whole Foods Market and Random House.Food52 has been named the best food website by the James Beard Foundation and IACP, and has been featured in the New York Times, NPR, Pando Daily, TechCrunch, and on the Today Show.We're located in Chelsea, in New York City. Food52, a fast-growing, James Beard Award-winning online food community and crowd-sourced and curated recipe hub, is currently interviewing full- and part-time unpaid interns to work in a small team of editors, executives, and developers in its New York City headquarters.Reproducing and/or repackaging existing Food52 content for a number of partner sites, such as Huffington Post, Yahoo, Buzzfeed, and more in their various content management systemsResearching blogs and websites for the Provisions by Food52 Affiliate ProgramAssisting in day-to-day affiliate program support, such as screening affiliates and assisting in any affiliate inquiriesSupporting with PR &amp; Events when neededHelping with office administrative work, such as filing, mailing, and preparing for meetingsWorking with developers to document bugs and suggest improvements to the siteSupporting the marketing and executive staff Experience with content management systems a major plus (any blogging counts!)Familiar with the Food52 editorial voice and aestheticLoves food, appreciates the importance of home cooking and cooking with the seasonsMeticulous editor, perfectionist, obsessive attention to detail, maddened by typos and broken links, delighted by finding and fixing themCheerful under pressureExcellent communication skillsA+ multi-tasker and juggler of responsibilities big and smallInterested in and engaged with social media like Twitter, Facebook, and PinterestLoves problem-solving and collaborating to drive Food52 forwardThinks big picture but pitches in on the nitty gritty of running a small company (dishes, shopping, administrative support)Comfortable with the realities of working for a startup: being on call on evenings and weekends, and working long hours   Other     Marketing|
+-------------+----------------+-------------+----------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
only showing top 1 row
"""
from pyspark.sql.functions import length
data = data.withColumn('length', length(data['text']))
data.show()

# In[2]: Feature Engineering
# IDF: Inverse Document Frequency
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer, VectorAssembler

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")#编译器
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')#停止词（是从输入中排除的词，通常是因为这些词经常出现，没有太多的含义）
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec', vocabSize=10000, minDF=2.0)#c_vec = term frequnce TF
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
fraudulent_to_label = StringIndexer(inputCol='fraudulent',outputCol='label')
final_feature = VectorAssembler(inputCols=['tf_idf', 'telecommuting','has_company_logo','has_questions'],outputCol='features')

from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages=[fraudulent_to_label,tokenizer,stopremove,count_vec,idf,final_feature])
clean_data = data_prep_pipe.fit(data).transform(data)
clean_data.show(1)
#clean_data.show(1,truncate=False)
"""
+-------------+----------------+-------------+----------+--------------------+------+-----+--------------------+--------------------+--------------------+--------------------+--------------------+
|telecommuting|has_company_logo|has_questions|fraudulent|                text|length|label|          token_text|         stop_tokens|               c_vec|              tf_idf|            features|
+-------------+----------------+-------------+----------+--------------------+------+-----+--------------------+--------------------+--------------------+--------------------+--------------------+
|            0|               1|            0|         0|Marketing Intern ...|  2710|  0.0|[marketing, inter...|[marketing, inter...|(262144,[0,1,2,3,...|(262144,[0,1,2,3,...|(262147,[0,1,2,3,...|
+-------------+----------------+-------------+----------+--------------------+------+-----+--------------------+--------------------+--------------------+--------------------+--------------------+
only showing top 1 row
"""


# In[5]: 
# ## Split data into training and test datasets
training, test = clean_data.randomSplit([0.8, 0.2], seed=12345)

# Build Logistic Regression Model
from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(featuresCol='features', labelCol='label')
logr_model = log_reg.fit(training)

results = logr_model.transform(test)
results.select('label','prediction').show()
results.show(5)
# In[6]:Evaluation
# #### Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
y_true = results.select("label")
y_true = y_true.toPandas()

y_pred = results.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred)
f1=f1_score(y_true, y_pred)
print(cnf_matrix)
print("Prediction Accuracy is ", (cnf_matrix[0,0]+cnf_matrix[1,1])/sum(sum(cnf_matrix)) )
print("True Positive Accuracy is ", (cnf_matrix[1,1]/ (cnf_matrix[1,1]+cnf_matrix[1,0])))
print("F1 Score is ",(f1))
