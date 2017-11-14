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
import spark.implicits._

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


// use OneHotEncoder to create dummy variables. OneHotEncoder can only encode numeric types - hence the indexer
val dow_indexer = new StringIndexer().setInputCol("pickup_dow").setOutputCol("pickup_dow_idx")
val dow_encoder = new OneHotEncoder().setInputCol("pickup_dow_idx").setOutputCol("pickup_dow_dummy")

// hour, minute
val min_encoder = new OneHotEncoder().setInputCol("pickup_minute").setOutputCol("pickup_minute_dummy")
val hour_encoder = new OneHotEncoder().setInputCol("pickup_hour").setOutputCol("pickup_hour_dummy")

// Still need to investigate depedency of variables to select features properly - esp. the month and day of month, and number of passengers.

val featureAssembler = new VectorAssembler()
  .setInputCols(Array(
      /*"pickup_month", "pickup_day",*/ "pickup_hour_dummy", "pickup_minute_dummy", "pickup_dow_dummy", 
      "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"))
  .setOutputCol("features")

val scaler = new StandardScaler()
  .setInputCol("features").setOutputCol("scaled_features")
  .setWithStd(true).setWithMean(false)
  
val lregression = new LinearRegression().setFeaturesCol("scaled_features").setLabelCol("trip_duration_min")

val randomForest = new RandomForestClassifier().setFeaturesCol(/*"scaled_features"*/"features").setLabelCol("trip_duration_min").setNumTrees(500)

// training pipeline
var trainPipe = new Pipeline().setStages(Array(dow_indexer, dow_encoder, min_encoder, hour_encoder, featureAssembler, /*scaler,*/ randomForest /*lregression*/))

// Training pipeline setup up to here - following lines are just for testing

// train, predict - using kaggleTrain and kaggleTest
var model = trainPipe.fit(kaggleTrain)
// Predict using model (PipeModel) - it contains the same transformations already as the pipeline that produced it
var p = model.transform(kaggleTest)

// train using the full dataset
var full_model = trainPipe.fit(fullDataSet)
var p_full = full_model.transform(kaggleTest)

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