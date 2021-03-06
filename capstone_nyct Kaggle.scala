// spark-shell

/*
 * Kaggle train data set
 */
val k_train = spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .option("mode", "DROPMALFORMED")
  .load("/user/bohdan/nyct/kaggle/train.csv")
  
/* register as table so you can use sql: */
k_train.createOrReplaceTempView("k_train")

// select columns, with new derived columns: month, day, dow, hour, minute for pickup, trip duration
var kaggleTrain = spark.sql("""
    select 
       passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
       /* derived columns */
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'MM') as Int) AS pickup_month,
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'DD') as Int) AS pickup_day,
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'HH') as Int) AS pickup_hour,
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'mm') as Int) AS pickup_minute,
       from_unixtime(unix_timestamp(pickup_datetime), 'EE') AS pickup_dow,
       trip_duration
    from k_train
    where trip_duration < 3600
""")

kaggleTrain.cache

/*
 * Kaggle test data set
 */
val k_test = spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .option("mode", "DROPMALFORMED")
  .load("/user/bohdan/nyct/kaggle/test-kaggle.csv")

k_test.createOrReplaceTempView("k_test")

// Columns as in the train set, minus the trip_duration
var kaggleTest = spark.sql("""
    select 
       passenger_count, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude,
       /* derived columns */
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'MM') as Int) AS pickup_month,
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'DD') as Int) AS pickup_day,
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'HH') as Int) AS pickup_hour,
       cast(from_unixtime(unix_timestamp(pickup_datetime,'MM/dd/yyyy'), 'mm') as Int) AS pickup_minute,
       from_unixtime(unix_timestamp(pickup_datetime), 'EE') AS pickup_dow
    from k_test
""")
kaggleTest.cache
