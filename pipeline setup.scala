/**
 * Only pipeline setup. Other parts of the setup are done in initial investigation script.
 *
 * Elements:     
 *               trip, a - dataframes
 *     feature_assembler - transformer
 *                    lr - estimator
 *                 model - transformer
**/

var pipeline = new Pipeline().setStages(Array(feature_assembler, lr))
var model = pipeline.fit(a)

// Not sure what these parameters mean - investigate
val paramGrid = new ParamGridBuilder()
//  .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .build()


val cvalidator = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(10)

var model = cvalidator.fit(a)