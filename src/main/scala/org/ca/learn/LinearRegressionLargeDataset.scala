package org.ca.learn
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator


object LinearRegressionLargeDataset {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegression")
      .getOrCreate()

    val data = spark.read.format("libsvm").load("s3a://sparkcookbook/housingdata/realestate.libsvm")
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))
    val lr = new LinearRegression()
    val model = lr.fit(training)
    val predictions = model.transform(test)
    val evaluator = new RegressionEvaluator()
    evaluator.evaluate(predictions)
  }
}