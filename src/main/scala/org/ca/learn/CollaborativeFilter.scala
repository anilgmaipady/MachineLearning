package org.ca.learn

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{Dataset, Encoders, SparkSession}

case class Rating(userId: Int, movieId: Int, rating: Double, timestamp: Long)

object CollaborativeFilter {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CollaborativeFilter")
      .getOrCreate()
    val encoder = Encoders.product[Rating]
    val ratings = spark.read.option("header", "true").option("inferschema", "true").csv("./src/main/scala/resource/ratings.csv").as[Rating](encoder).cache()
    val Array(training, test) = ratings.randomSplit(Array(0.7, 0.3))
    val als = new ALS().setMaxIter(30).setRegParam(.065).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
    val model = als.fit(training)
    val predictions = model.transform(test).na.drop
    predictions.columns.foreach(println)
    println(predictions.first())
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
  }
}
