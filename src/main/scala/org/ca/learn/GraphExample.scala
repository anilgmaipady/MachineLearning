package org.ca.learn

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object GraphExample {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("GraphExample Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val vertices = Array((1L, ("Santa Clara", "CA")), (2L, ("Fremont", "CA")), (3L, ("San Francisco", "CA")))
    val vrdd = sc.parallelize(vertices)
    val edges = Array(Edge(1L,2L,20),Edge(2L,3L,44),Edge(3L,1L,53))
    val erdd = sc.parallelize(edges)
    val graph = Graph(vrdd,erdd)
    println("Vertices:")
    graph.vertices.collect.foreach(println)
    println("Edges:")
    graph.edges.collect.foreach(println)
    println("Triplets:")
    graph.triplets.collect.foreach(println)
    println("Number of inward-directed edges:")
    graph.inDegrees.foreach(println)
  }
}
