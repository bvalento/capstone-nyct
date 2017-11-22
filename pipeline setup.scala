/**
 * Only pipeline setup. Other parts of the setup are done in initial investigation script.
 *
 * Elements:     
 *           
 *           dow_indexer - Day-of-week indexer (turns day of week name into factor) - see if dow_encoder can work directly with dow without this step
 *           *_encoder   - One-hot-encoders (dummy var) for day of week
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

// use OneHotEncoder to create dummy variables
val hour_encoder = new OneHotEncoder().setInputCol("pickup_hour").setOutputCol("pickup_hour_dummy")
val month_encoder = new OneHotEncoder().setInputCol("pickup_month").setOutputCol("pickup_month_dummy")
val psgr_encoder = new OneHotEncoder().setInputCol("passenger_count").setOutputCol("passenger_count_dummy")

val featureAssembler = new VectorAssembler()
  .setInputCols(Array(
      "pickup_month_dummy", "pickup_hour_dummy", "passenger_count",
      "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"))
  .setOutputCol("features")

val scaler = new StandardScaler()
  .setInputCol("features").setOutputCol("scaled_features")
  .setWithStd(true).setWithMean(false)
  
val lregression = new LinearRegression().setFeaturesCol("scaled_features").setLabelCol("trip_duration").setRegParam(0.3).setMaxIter(10)

// training pipeline
var trainPipe = new Pipeline().setStages(Array(hour_encoder, month_encoder, featureAssembler, scaler, lregression))

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
