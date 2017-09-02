package org.ca.learn

import org.apache.spark.sql.SparkSession
import org.graphframes._

object DataFrameRoadSample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("LinearRegression")
      .getOrCreate()

    //https://sparkcookbook.s3.amazonaws.com/roads/ca/roadNet-CA.txt
    val edges = spark.read.option("delimiter","\t").option("header","true").option("inferschema","true").csv("./src/main/resources/roadNet-CA-small.txt")
    val vertices = edges.select("src").union(edges.select("dst")).distinct.toDF("id")

    val g = GraphFrame(vertices, edges)

    g.vertices.show

    g.edges.show

    g.inDegrees.show

    spark.sparkContext.setCheckpointDir("/tmp/");

    val cc = g.connectedComponents.run

    cc.printSchema

    val comp = cc.select("component").distinct.count

    println("Clusters:")
    print(comp)

  }

}
