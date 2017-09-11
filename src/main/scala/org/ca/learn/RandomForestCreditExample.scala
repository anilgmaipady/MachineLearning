package org.ca.learn

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object RandomForestCreditExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RandomForestExample")
      .getOrCreate()
    //Label Generosity:Responsibility:Care:Organization:Spendthrift:Volatile
    val data = spark.read.format("libsvm").load("./src/main/resources/rf-credit.libsvm")
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))
    val rf = new RandomForestClassifier().setNumTrees(3)
    val model = rf.fit(training)

    val predictions = model.transform(test)
    predictions.show(20,false)

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    print("Accuracy:")
    println(accuracy)
    print("Model")
    println(model.toDebugString)

    val test2 = spark.createDataFrame(Seq(Vectors.dense(0,0,0,0,1,1)).map(Tuple1.apply)).toDF("features")
    val predictions2 = model.transform(test2)
    predictions2.show(1,false)


    val test3 = spark.createDataFrame(Seq(Vectors.dense(1,1,1,1,0,0)).map(Tuple1.apply)).toDF("features")
    val predictions3 = model.transform(test3)
    predictions3.show(1,false)


  }
}
