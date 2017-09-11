package org.ca.learn

import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

object GradientBoostedTreesExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RandomForestExample")
      .getOrCreate()
    /*
    Class variable (0 or 1)
    Number of times pregnant
    Plasma glucose concentration at 2 hours in an oral glucose tolerance test
    Diastolic blood pressure (mmHg)
    Triceps skinfold thickness (mm)
    2-hour serum insulin (mu U/ml)
    Body mass index (weight in kg/(height in m2 )
    Diabetes pedigree function
    Age (years)
    */
    val data = spark.read.format("libsvm").load("./src/main/resources/diabetes.libsvm")
    val Array(training, test) = data.randomSplit(Array(0.7, 0.3))
    val gbt = new GBTClassifier().setMaxIter(10)
    val model = gbt.fit(training)

    val predictions = model.transform(test)
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    print("Accuracy:")
    println(accuracy)
    print("Model")
    println(model.toDebugString)
  }
}
