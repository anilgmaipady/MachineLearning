package org.ca.learn

import org.apache.spark.sql.SparkSession
import org.graphframes._

object DataFrameSimpleSample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegression")
      .getOrCreate()

    val vertices = spark.sqlContext.createDataFrame(List(
      ("sc", "Santa Clara", "CA"),
      ("fr", "Fremont", "CA"),
      ("sf", "San Francisco", "CA")))
      .toDF("id", "city", "state")


    val edges = spark.sqlContext.createDataFrame(List(
      ("sc","fr",20),
      ("fr","sf",44),
      ("sf","sc",53)))
      .toDF("src","dst","distance")


    val g = GraphFrame(vertices, edges)

    g.vertices.show

    g.edges.show

    g.inDegrees.show


  }

}
