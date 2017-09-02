package org.ca.learn

import org.apache.log4j.LogManager
import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.graphx._

object NeighborhoodAggregation {
  def main(args: Array[String]): Unit = {
    val log = LogManager.getRootLogger
    val conf = new SparkConf().setAppName("GraphExample Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val edgesFile = sc.textFile("./src/main/resources/edges_neigh.csv")
    val edges = edgesFile.map(_.split(",")).map(e => Edge(e(0).toLong,e(1).toLong,e(2)))
    val verticesFile = sc.textFile("./src/main/resources/nodes_neigh.csv")
    val vertices = verticesFile.map(_.split(",")).map( e => (e(0).toLong,e(1)))
    val graph = Graph(vertices,edges)
    log.warn("Graph:")
    graph.triplets.collect.foreach(log.warn)
    val followerCount = graph.aggregateMessages[(Int)]( t => t.sendToDst(1), (a, b) => (a+b))
    log.warn("FollowerCount:")
    followerCount.collect.foreach(log.warn)

  }
}
