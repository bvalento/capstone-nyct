// spark-shell

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import spark.implicits._

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Function to convert days to factor
spark.sqlContext.udf.register("dayToInt", (dayOfWeek:String) => {
    dayOfWeek.toLowerCase() match {
        case "sunday"    => 1
        case "monday"    => 2
        case "tuesday"   => 3 
        case "wednesday" => 4
        case "thursday"  => 5
        case "friday"    => 6
        case "saturday"  => 7
    }
})

/*
 * Data frame with SQL - with a small subset test.csv
 */
val trips = spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .option("mode", "DROPMALFORMED")
  .load("/user/bohdan/nyct/test.csv")
/*  .load("/user/bohdan/nyct/yellow_tripdata_2016-01-06.csv") */
  
/* register as table so you can use sql: */
trips.createOrReplaceTempView("trips")

// now the data frame can be queried using sql:
spark.sql("select * from trips")

trips.select("VendorID").distinct.collect

// select columns, with new derived columns: month, day, dow, hour, minute for pickup, trip duration
var a = spark.sql("""
    select tpep_pickup_datetime, tpep_dropoff_datetime, 
       passenger_count,    
       pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
       /* derived columns */
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'MM') as Int) AS pickup_month,
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'DD') as Int) AS pickup_day,
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'HH') as Int) AS pickup_hour,
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'mm') as Int) AS pickup_minute,
       from_unixtime(unix_timestamp(tpep_pickup_datetime), 'EE') AS pickup_dow,
       /* trip duration in seconds */
       unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime) AS trip_duration
    from trips
    where pickup_longitude!=0 and pickup_latitude!=0 and dropoff_longitude!=0 and dropoff_latitude!=0
""")

// some example funcions on data frame
a.select("pickup_dow").distinct.collect
a.groupBy("pickup_dow").count.show

// use pipe elements directly, bypassing the pipe (just to try things out). All elements are set up in "pipeline setup.scala", used here.

val indexed = dow_indexer.transform(a)

// now use OneHotEncoder to create dummy variables
val encoded = dow_encoder.transform(indexed)

// select features
val featurized = feature_assembler.transform(encoded)

/* 
 * Linear regression model. Just for testing - will be replaced with the pipeline
 * that is for now in a separate file
 */
val lregression = new LinearRegression().setFeaturesCol("features").setLabelCol("trip_duration")
var model = lregression.fit(featurized)
println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

// RDD func, might not work with data frames
trips.filter(!_startsWith("VendorID,tpep_pickup_datetime,")).count

// various util functions - examples


jan16.printSchema
jan16.select("VendorID").distinct()
jan16.groupBy("VendorID").count().show()