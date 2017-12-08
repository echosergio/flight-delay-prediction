import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.SparkSession

object App {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Flight Delay Prediction")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    import spark.implicits._

    val initialFlightDf = spark
      .read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ",")
      .option("nullValue", "NA")
      .option("mode", "DROPMALFORMED")
      .load(args(0))

    val flightDf = initialFlightDf
      .drop("ArrTime")
      .drop("ActualElapsedTime")
      .drop("AirTime")
      .drop("TaxiIn")
      .drop("Diverted")
      .drop("CarrierDelay")
      .drop("WeatherDelay")
      .drop("NASDelay")
      .drop("SecurityDelay")
      .drop("LateAircraftDelay")
      // Unnecessary and non quantitative variables
      .drop("TailNum")
      .drop("UniqueCarrier")
      .drop("Cancelled")
      .drop("CancellationCode")
      .na.drop
      // Cast variables
      .select(
        col("Year").cast(IntegerType),
        col("Month").cast(IntegerType),
        col("DayofMonth").cast(IntegerType),
        col("DayOfWeek").cast(IntegerType),
        col("DepTime").cast(IntegerType),
        col("CRSDepTime").cast(IntegerType),
        col("CRSArrTime").cast(IntegerType),
        col("FlightNum").cast(IntegerType),
        col("CRSElapsedTime").cast(IntegerType),
        col("ArrDelay").cast(IntegerType),
        col("DepDelay").cast(IntegerType),
        col("Origin").cast(StringType),
        col("Dest").cast(StringType),
        col("Distance").cast(IntegerType),
        col("TaxiOut").cast(IntegerType)
      )

    val Array(trainingData, testData) = flightDf.randomSplit(Array(0.7, 0.3))

    val originIndexer = new StringIndexer().setInputCol("Origin").setOutputCol("OriginIndex").setHandleInvalid("skip")
    val destIndexer = new StringIndexer().setInputCol("Dest").setOutputCol("DestIndex").setHandleInvalid("skip")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("Year", "Month", "DayofMonth", "DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "FlightNum",
        "CRSElapsedTime", "ArrDelay", "DepDelay", "OriginIndex", "DestIndex", "Distance", "TaxiOut"))
      .setOutputCol("rawFeatures")

    val vectorSlicer = new VectorSlicer()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .setNames(Array("DayOfWeek", "DepTime", "CRSDepTime", "CRSArrTime", "CRSElapsedTime", "DepDelay", "OriginIndex", "DestIndex", "Distance", "TaxiOut"))

    val linearRegression = new LinearRegression()
      .setLabelCol("ArrDelay")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.3) // Increasing lambda results in less overfitting but also greater bias
      .setElasticNetParam(0.8) // Reduce overfitting

    val pipeline = new Pipeline().setStages(Array(originIndexer, destIndexer, vectorAssembler, vectorSlicer, linearRegression))

    // Cross-validation

    val paramGrid = new ParamGridBuilder()
      .addGrid(linearRegression.regParam, Array(0.1, 0.5, 1))
      .addGrid(linearRegression.elasticNetParam, Array(0.1, 0.5, 1))
      .build()

    val regressionEvaluator = new RegressionEvaluator()
      .setLabelCol("ArrDelay")
      .setPredictionCol("prediction")
      .setMetricName("r2")

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(regressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)

    val crossValidatorModel = crossValidator.fit(trainingData)
    val predictions = crossValidatorModel.transform(testData)

    predictions.select("features", "ArrDelay", "prediction").show()
    println("Accuracy: " + regressionEvaluator.evaluate(predictions))

  }

}
