package org.ca.learn

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession


object LogisticRegressionFootBallPlayers {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegression")
      .getOrCreate()

    //height in inches and weight in lb
    val lebron = (1.0,Vectors.dense(80.0,250.0))
    val tim = (0.0,Vectors.dense(70.0,150.0))
    val brittany = (1.0,Vectors.dense(80.0,207.0))
    val stacey = (0.0,Vectors.dense(65.0,120.0))
    val trainingDataSet = spark.createDataFrame(Seq(lebron,tim,brittany,stacey)).toDF("label","features")



    val lr = new LogisticRegression()
    val model = lr.fit(trainingDataSet)


    val john = Vectors.dense(90.0,270.0)
    val tom = Vectors.dense(62.0,120.0)

    val testData = spark.createDataFrame(Seq((1.0, john), (0.0, tom))).toDF("label", "features")

    val results = model.transform(testData)


    results.printSchema()

    results.show()

  }
}