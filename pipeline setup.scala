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

// use OneHotEncoder to create dummy variables. OneHotEncoder can only encode numeric types - hence the indexer
val dow_indexer = new StringIndexer().setInputCol("pickup_dow").setOutputCol("pickup_dow_idx")
val dow_encoder = new OneHotEncoder().setInputCol("pickup_dow_idx").setOutputCol("pickup_dow_dummy")

// hour, minute
val min_encoder = new OneHotEncoder().setInputCol("pickup_minute").setOutputCol("pickup_minute_dummy")
val hour_encoder = new OneHotEncoder().setInputCol("pickup_hour").setOutputCol("pickup_hour_dummy")

// Still need to investigate depedency of variables to select features properly - esp. the month and day of month, and number of passengers.

val feature_assembler = new VectorAssembler()
  .setInputCols(Array(
      /*"pickup_month", "pickup_day",*/ "pickup_hour_dummy", "pickup_minute_dummy", "pickup_dow_dummy", 
      "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"))
  .setOutputCol("features")

val lregression = new LinearRegression().setFeaturesCol("features").setLabelCol("trip_duration")

var pipeline = new Pipeline().setStages(Array(dow_indexer, dow_encoder, min_encoder, hour_encoder, feature_assembler, lregression))


var model = pipeline.fit(a)

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