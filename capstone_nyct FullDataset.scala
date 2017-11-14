// spark-shell

/*
 * Data frame with SQL - with a small subset test.csv
 */
val trips = spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .option("mode", "DROPMALFORMED")
/*  .load("/user/bohdan/nyct/test.csv") */
  .load("/user/bohdan/nyct/yellow_tripdata_2016-01-06.csv")
  
/* register as table so you can use sql: */
trips.createOrReplaceTempView("trips")

// now the data frame can be queried using sql:
spark.sql("select * from trips")

trips.select("VendorID").distinct.collect

// select columns, with new derived columns: month, day, dow, hour, minute for pickup, trip duration
var fullDataSet = spark.sql("""
    select tpep_pickup_datetime, tpep_dropoff_datetime, passenger_count,    
       pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
       /* derived columns */
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'MM') as Int) AS pickup_month,
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'DD') as Int) AS pickup_day,
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'HH') as Int) AS pickup_hour,
       cast(from_unixtime(unix_timestamp(tpep_pickup_datetime,'MM/dd/yyyy'), 'mm') as Int) AS pickup_minute,
       from_unixtime(unix_timestamp(tpep_pickup_datetime), 'EE') AS pickup_dow,
       /* trip duration in seconds */
       unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime) AS trip_duration,
       cast((unix_timestamp(tpep_dropoff_datetime) - unix_timestamp(tpep_pickup_datetime)) / 60 AS Int) AS trip_duration_min
    from trips
    where pickup_longitude!=0 and pickup_latitude!=0 and dropoff_longitude!=0 and dropoff_latitude!=0 AND tpep_pickup_datetime != tpep_dropoff_datetime
""")

val splitSeed = 5043
val Array(fullTrain, fullTest) = fullDataSet.randomSplit(Array(0.7, 0.3), splitSeed)

fullTrain.cache
fullTest.cache