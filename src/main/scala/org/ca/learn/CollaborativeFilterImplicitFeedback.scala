package org.ca.learn

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{Dataset, Encoders, SparkSession}

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.RegressionEvaluator

object CollaborativeFilterImplicitFeedback {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("CollaborativeFilter")
      .getOrCreate()
    val encoder = Encoders.product[Rating]
    val als = new ALS().setImplicitPrefs(true).setUserCol("userId").setItemCol("songId").setRatingCol("plays")
    val songs = spark.read.option("delimiter"," ").option("inferschema","true").csv("./src/main/resources/kaggle_songs.txt").toDF("song","songId")
    songs.show()
    songs.createOrReplaceTempView("songs")
    val users = spark.read.textFile("./src/main/scala/resource/kaggle_users.txt").toDF("user").withColumn("userId",monotonically_increasing_id)
    users.show()
    val u =  users.withColumn("userId",users.col("userId").cast("integer"))
    u.createOrReplaceTempView("users")
    val triplets1 = spark.read.option("delimiter",",").option("inferschema","true").csv("./src/main/resources/kaggle_visible_evaluation_triplets.txt")
    triplets1.show()
    val triplets  = triplets1.toDF("user","song","plays")
    val t = triplets.withColumn("plays",triplets.col("plays").cast("double"))
    t.createOrReplaceTempView("plays")
    val plays = spark.sql("select userId,songId,plays from plays p join users u on p.user = u.user join songs s on p.song = s.song")
    val Array(training, test) = plays.randomSplit(Array(0.7, 0.3))
    val model = als.fit(training)
    val predictions = model.transform(test).na.drop
    val evaluator = new RegressionEvaluator().setMetricName("mae").setLabelCol("plays").setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

  }
}
