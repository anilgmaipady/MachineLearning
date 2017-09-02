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
    //https://sparkcookbook.s3.amazonaws.com/moviedata/ratings.csv
    val ratings = spark.read.option("header", "true").option("inferschema", "true").csv("./src/main/scala/resource/ratings-do-not-commit.csv").as[Rating](encoder).cache()
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


    val myrecs = spark.createDataFrame(Seq(
      (138494,1721,5,1489789319),
      (138494,10,3,1489789319),
      (138494,1,1,1489789319),
      (138494,225,4,1489789319),
      (138494,344,4,1489789319),
      (138494,480,5,1489789319),
      (138494,589,5,1489789319),
      (138494,780,4,1489789319),
      (138494,1049,4,1489789319)
    )).toDF("userId","movieId","rating","timestamp").as[Rating](encoder)

    val trainingWithMyRecs = training.union(myrecs)

    val model2 = als.fit(trainingWithMyRecs)

    val terminator = spark.createDataFrame(Seq((138494,1240) )).toDF("userId","movieId")

    val p = model2.transform(terminator)

    p.show()


  }
}
