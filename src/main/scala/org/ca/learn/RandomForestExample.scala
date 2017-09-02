package org.ca.learn

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SparkSession

object RandomForestExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegression")
      .getOrCreate()
    val data = spark.read.format("libsvm").load("./src/main/resources/diabetes.libsvm")
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))
    val rf = new RandomForestClassifier().setNumTrees(3)
    val model = rf.fit(training)
    val predictions = model.transform(test)
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    print("Accuracy:")
    println(accuracy)
    print("Model")
    println(model.toDebugString)
  }
}
