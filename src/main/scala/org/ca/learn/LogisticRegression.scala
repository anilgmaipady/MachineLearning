package org.ca.learn

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession



import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.linalg.{Vector, Vectors}


object LogisticRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegression")
      .getOrCreate()
    val trainingDataSet = spark.createDataFrame(Seq(
      (0.0,Vectors.dense(0.245)),
      (0.0,Vectors.dense(0.247)),
      (1.0,Vectors.dense(0.285)),
      (1.0,Vectors.dense(0.299)),
      (1.0,Vectors.dense(0.327)),
      (1.0,Vectors.dense(0.347)),
      (0.0,Vectors.dense(0.356)),
      (1.0,Vectors.dense(0.36)),
      (0.0,Vectors.dense(0.363)),
      (1.0,Vectors.dense(0.364)),
      (0.0,Vectors.dense(0.398)),
      (1.0,Vectors.dense(0.4)),
      (0.0,Vectors.dense(0.409)),
      (1.0,Vectors.dense(0.421)),
      (0.0,Vectors.dense(0.432)),
      (1.0,Vectors.dense(0.473)),
      (1.0,Vectors.dense(0.509)),
      (1.0,Vectors.dense(0.529)),
      (0.0,Vectors.dense(0.561)),
      (0.0,Vectors.dense(0.569)),
      (1.0,Vectors.dense(0.594)),
      (1.0,Vectors.dense(0.638)),
      (1.0,Vectors.dense(0.656)),
      (1.0,Vectors.dense(0.816)),
      (1.0,Vectors.dense(0.853)),
      (1.0,Vectors.dense(0.938)),
      (1.0,Vectors.dense(1.036)),
      (1.0,Vectors.dense(1.045)))).toDF("label","features")
    val lr = new LogisticRegression()
    val model = lr.fit(trainingDataSet)
    val trainingSummary = model.summary

    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]


    /*
    ROC is a statistical tool to assess the accuracy of predictions. The accuracy of predictions plays a major role in how predictions would be used.
    Quoting Prof. Andrew Ng, Stanford University, chief scientist at Baidu, here for reference:
    "The difference between 95% accuracy versus 99% accuracy is between you sometimes making use of it versus using it all the time ."
    ROC
    */

    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    val test = spark.createDataFrame(Seq(Vectors.dense(1.045)).map(Tuple1.apply)).toDF("features")
    val predictions = model.transform(test)

    predictions.show()

  }
}