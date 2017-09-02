package org.ca.learn

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.model.DecisionTreeModel

object TryDecisionTree {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val data = sc.textFile("./src/main/resources/tennis.csv")
    val parsedData = data.map {
      line =>  val parts = line.split(',').map(_.toDouble)
        LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }
    val model = DecisionTree.train(parsedData, Classification, Entropy, 3)
    val v=Vectors.dense(0.0,1.0,0.0)
    val decisionTreeModel = model.predict(v)
    print("Decision:")
    println(decisionTreeModel)
    println(model.toDebugString)
  }
}
