/**
 * Only pipeline setup. Other parts of the setup are done in initial investigation script.
 *
 * Elements:     
 *               trip, a - dataframes
 *           dow_indexer - Day-of-week indexer (turns day of week name into factor) - see if dow_encoder can work directly with dow without this step
 *           dow_encoder - One-hot-encoder (dummy var) for day of week
 *     feature_assembler - transformer
 *           lregression - estimator
 *                 model - transformer
**/

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

import org.apache.spark.mllib.evaluation.RegressionMetrics

import spark.implicits._

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


// use OneHotEncoder to create dummy variables. OneHotEncoder can only encode numeric types - hence the indexer
val dow_indexer = new StringIndexer().setInputCol("pickup_dow").setOutputCol("pickup_dow_idx")
val dow_encoder = new OneHotEncoder().setInputCol("pickup_dow_idx").setOutputCol("pickup_dow_dummy")

// hour, minute
val min_encoder = new OneHotEncoder().setInputCol("pickup_minute").setOutputCol("pickup_minute_dummy")
val hour_encoder = new OneHotEncoder().setInputCol("pickup_hour").setOutputCol("pickup_hour_dummy")

// string indexer for label - needed by the random tree forest
val labelIndexer = new StringIndexer().setInputCol("trip_duration").setOutputCol("trip_duration_idx")


// Still need to investigate depedency of variables to select features properly - esp. the month and day of month, and number of passengers.

val featureAssembler = new VectorAssembler()
  .setInputCols(Array(
      "pickup_month", /*"pickup_day",*/ "pickup_hour_dummy", /*"pickup_minute_dummy", "pickup_dow_dummy",*/ 
      "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"))
  .setOutputCol("features")

val scaler = new StandardScaler()
  .setInputCol("features").setOutputCol("scaled_features")
  .setWithStd(true).setWithMean(false)
  
val lregression = new LinearRegression().setFeaturesCol("scaled_features" /*"features"*/).setLabelCol("trip_duration")

//val randomForest = new RandomForestClassifier().setFeaturesCol(/*"scaled_features"*/"features").setLabelCol("trip_duration_idx").setNumTrees(10)

// training pipeline
var trainPipe = new Pipeline().setStages(Array(dow_indexer, dow_encoder, min_encoder, hour_encoder, featureAssembler, scaler, labelIndexer, lregression))

// Training pipeline setup up to here - following lines are just for testing

// Create two models, one for kaggle dataset, one for the 95% of full data set
var kaggleModel = trainPipe.fit(kaggleTrain)
var fullModel = trainPipe.fit(fullTrain)

// save models (saves them to HDFS)
kaggleModel.write.overwrite().save("/home/bohdan/kaggleModel_featSelected")
fullModel.write.overwrite().save("/home/bohdan/fullModel_featSelected")

/*
 * To load the models: 
 *    var kaggleModel = PipelineModel.load("/home/bohdan/kaggleModel")
 *    var fullModel = PipelineModel.load("/home/bohdan/fullModel")
 */

// Predict using model (PipeModel) - it contains the same transformations already as the pipeline that produced it
var p_kaggle = kaggleModel.transform(fullTest)
var p_full = fullModel.transform(fullTest)

// save actual values and predictions as csv
p_kaggle.select("trip_duration", "prediction").coalesce(1).write.format("csv").option("header", "true").save("/home/bohdan/kaggle_predictions.csv")
p_full  .select("trip_duration", "prediction").coalesce(1).write.format("csv").option("header", "true").save("/home/bohdan/full_predictions.csv")

// Look at the metrics for both predictions. First change the data frames to RDDs, with actual and predicted values
// for some reason when using rdd vars from above, Spark doesn't like the Long on the position 0. This way it works (perhaps investigate later why)
var metricsKaggle = new RegressionMetrics(p_kaggle.select("trip_duration", "prediction").rdd.map(l => (l.getLong(0), l.getDouble(1))))
var metricsFull = new RegressionMetrics(p_full.select("trip_duration", "prediction").rdd.map(l => (l.getLong(0), l.getDouble(1))))

// Squared error
println(s"MSE:  Kaggle: ${metricsKaggle.meanSquaredError}, Full data set: ${metricsFull.meanSquaredError}")
println(s"RMSE: Kaggle: ${metricsKaggle.rootMeanSquaredError}, Full data set: ${metricsFull.rootMeanSquaredError}")

// R-squared
println(s"R-squared: Kaggle: ${metricsKaggle.r2}, Full data set: ${metricsFull.r2}")

// Mean absolute error
println(s"MAE: Kaggle: ${metricsKaggle.meanAbsoluteError}, Full data set: ${metricsFull.meanAbsoluteError}")

// Explained variance
println(s"Explained variance: Kaggle: ${metricsKaggle.explainedVariance}, Full data set: ${metricsFull.explainedVariance}")


// Not sure what these parameters mean - investigate
val paramGrid = new ParamGridBuilder()
//  .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
  .addGrid(lregression.regParam, Array(0.1, 0.01))
  .build()


val cvalidator = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(10)

var pipemodel = cvalidator.fit(a)