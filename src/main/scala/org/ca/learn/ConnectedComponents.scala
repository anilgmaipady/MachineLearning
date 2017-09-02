package org.ca.learn

import org.apache.log4j.LogManager
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx._

object ConnectedComponents {
  def main(args: Array[String]): Unit = {
    val log = LogManager.getRootLogger
    val conf = new SparkConf().setAppName("ConnectedComponents Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val edgesFile = sc.textFile("./src/main/resources/edges.csv")
    val edges = edgesFile.map(_.split(",")).map(e => Edge(e(0).toLong,e(1).toLong,e(2)))
    val verticesFile = sc.textFile("./src/main/resources/nodes.csv")
    val vertices = verticesFile.map(_.split(",")).map( e => (e(0).toLong,e(1)))
    val graph = Graph(vertices,edges)
    graph.triplets.collect().foreach(log.info)
    val cc = graph.connectedComponents
    log.info("connectedComponents:")
    cc.triplets.collect.foreach(log.info)

    val ccVertices = cc.vertices

    ccVertices.collect.foreach(println)

    val edges1 = cc.edges

    edges1.collect.foreach(println)

  }
}
