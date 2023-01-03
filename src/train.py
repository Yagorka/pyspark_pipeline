MASTER = "local"
NUM_PROCESSORS = "8"
NUM_EXECUTORS = "4"
NUM_PARTITIONS = 10

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.sql import SQLContext
import pyspark.sql.functions as psf
import json

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame
from typing import Iterable

conf = SparkConf()

conf.set("spark.app.name", "one_part_data")
conf.set("spark.master", MASTER)
conf.set("spark.executor.cores", NUM_PROCESSORS)
conf.set("spark.executor.instances", NUM_EXECUTORS)
conf.set("spark.executor.memory", "6g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer.max", "2000")
conf.set("spark.executor.heartbeatInterval", "6000s")
conf.set("spark.network.timeout", "10000000s")
conf.set("spark.shuffle.spill", "true")
conf.set("spark.driver.memory", "15g")
conf.set("spark.driver.maxResultSize", "15g")

from pyspark.sql import SparkSession

# Create SparkSession 
spark = SparkSession.builder \
      .config(conf=conf) \
      .master("local[*]") \
      .appName("SparkByExamples.com") \
      .getOrCreate()
spark

dataset = spark.read.csv("/home/yagor/Рабочий стол/mipt/lab3/notebook/nutrition_table.csv",header=True,inferSchema=True)

feat_cols = [ #'_c0',
 'fat_100g',
 'carbohydrates_100g',
 'sugars_100g',
 'proteins_100g',
 'salt_100g',
 'energy_100g',
 'reconstructed_energy',
 'g_sum',
 'exceeded',
 #'product'
 ]

# CUSTOM TRANSFORMER ----------------------------------------------------------------
class ColumnDropper(Transformer):
    """
    A custom Transformer which drops all columns that have at least one of the
    words from the banned_list in the name.
    """
    def __init__(self, banned_list: Iterable[str]):
        super(ColumnDropper, self).__init__()
        self.banned_list = banned_list
    def _transform(self, df: DataFrame) -> DataFrame:
        df = df.drop(*[x for x in df.columns if any(y in x for y in self.banned_list)])
        return df

column_dropper = ColumnDropper(banned_list = ['rawPrediction', 'probability'])

vec_assembler = VectorAssembler(inputCols = feat_cols, outputCol='features')
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

k = 2

kmeans = KMeans(featuresCol='scaledFeatures', predictionCol='pred_kmeans_cluster', k=k)

RandomForest = RandomForestClassifier(labelCol="pred_kmeans_cluster", featuresCol="scaledFeatures", \
                                      predictionCol='pred_from_randomforest_class', numTrees=20, maxDepth=3)

LogRegression = LogisticRegression(maxIter=10, regParam=0.1, elasticNetParam=1.0, labelCol="pred_from_randomforest_class", \
                                   featuresCol="scaledFeatures", probabilityCol="lr_prob", predictionCol='pred_from_logregression_class')

(trainingData, testData) = dataset.randomSplit([0.7, 0.3])

print(f'Metric for models on test data with {testData.count()} rows, train data {trainingData.count()} rows.')

pipeline = Pipeline(stages=[vec_assembler, scaler, kmeans, RandomForest, column_dropper, LogRegression])

model = pipeline.fit(trainingData)
output = model.transform(testData)

evaluator = ClusteringEvaluator(predictionCol='pred_kmeans_cluster', featuresCol='scaledFeatures')
silhouette = evaluator.evaluate(output)
d = 50
print('-'*d)

print(f'Metric for KMeans:')
print(f"With k={k} Silhouette with squared euclidean distance = " + str(silhouette))
output.groupBy('pred_kmeans_cluster').count().show()

print('-'*d)
print()
print('-'*d)

print(f'Metric for RandomForestClassifier:')
# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="pred_kmeans_cluster", predictionCol="pred_from_randomforest_class", metricName="accuracy")
accuracy = evaluator.evaluate(output)
print("Test Error = %g" % (1.0 - accuracy))

y_true = output.select(['pred_kmeans_cluster']).collect()
y_pred = output.select(['pred_from_randomforest_class']).collect()

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred))

print('-'*d)
print()
print('-'*d)

print(f'Metric for RandomForestClassifier:')
# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(model.stages[5].coefficientMatrix))
print("Intercept: " + str(model.stages[5].interceptVector))

trainingSummary = model.stages[5].summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))

print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
print('-'*d)

