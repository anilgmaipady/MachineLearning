package org.ca.learn

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.model.DecisionTreeModel

object DecisionTreeExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    //play(Yes:1.0 No:0.0) rain(yes:2.0 no:1.0),windy(yes:2.0 no:1.0),temperature(hot:3.0,normal:2.0,cool:1.0)
    val data = sc.textFile("./src/main/resources/tennis-modified.csv")
    val parsedData = data.map {
      line =>  val parts = line.split(',').map(_.toDouble)
        LabeledPoint(parts(0), Vectors.dense(parts.tail))
    }
    val model = DecisionTree.train(parsedData, Classification, Entropy, 3)
    val v=Vectors.dense(2.0,2.0,3.0)
    val decisionTreeModel = model.predict(v)
    print("Decision:")
    println(decisionTreeModel)
    println(model.toDebugString)
  }
}
