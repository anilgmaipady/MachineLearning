package org.ca.learn

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.SVMWithSGD

object SVM {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)

    //val data = MLUtils.loadLibSVMFile(sc,"s3a://sparkcookbook/medicaldata/diabetes.libsvm")
    /*
    Number of times pregnant
    Plasma glucose concentration at 2 hours in an oral glucose tolerance test
    Diastolic blood pressure (mmHg)
    Triceps skinfold thickness (mm)
    2-hour serum insulin (mu U/ml)
    Body mass index (weight in kg/(height in m2 )
    Diabetes pedigree function
    Age (years)
    Class variable (0 or 1)
     */

    val data = MLUtils.loadLibSVMFile(sc,"./src/main/resources/diabetes.libsvm")
    println(data.count)
    val trainingAndTest = data.randomSplit(Array(0.5,0.5))
    val trainingData = trainingAndTest(0)
    val testData = trainingAndTest(1)
    val model = SVMWithSGD.train(trainingData,100)
    val label = model.predict(testData.first.features)
    println(label)
    val predictionsAndLabels = testData.map( r => (model.predict(r.features),r.label))
    predictionsAndLabels.collect().foreach(println)
    println(predictionsAndLabels.filter(p => p._1 != p._2).count)
  }
}