package org.ca.learn

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession


import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression

object LassoLinearRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegression")
      .getOrCreate()

    val points = spark.createDataFrame(Seq(
      (1d,Vectors.dense(5,3,1,2,1,3,2,2,1)),
      (2d,Vectors.dense(9,8,8,9,7,9,8,7,9))
    )).toDF("label","features")
    val lr = new LinearRegression().setMaxIter(10).setRegParam(.3).setFitIntercept(false).setElasticNetParam(1.0)
    val model = lr.fit(points)
    println(model.coefficients)

    val test = spark.createDataFrame(Seq(Vectors.dense(1,1,1,1,1,1,1,1,1)).map(Tuple1.apply)).toDF("features")
    val predictions = model.transform(test)
    predictions.collect().foreach(println)


    var test2 = spark.createDataFrame(Seq(Vectors.dense(9,8,8,9,7,9,8,7,9)).map(Tuple1.apply)).toDF("features")
    var predictions2 = model.transform(test2)
    predictions2.collect().foreach(println)

    var test3 = spark.createDataFrame(Seq(Vectors.dense(5,3,1,2,1,3,2,2,1)).map(Tuple1.apply)).toDF("features")
    var predictions3 = model.transform(test3)
    predictions3.collect().foreach(println)

  }
}