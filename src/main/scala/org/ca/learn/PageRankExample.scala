package org.ca.learn

import org.apache.log4j.LogManager
import org.apache.spark.graphx._
import org.apache.spark.{SparkConf, SparkContext}

object PageRankExample {
  def main(args: Array[String]): Unit = {
    val log = LogManager.getRootLogger
    val conf = new SparkConf().setAppName("GraphExample Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val edgesFile = sc.textFile("./src/main/resources/links_small.txt", 20)
    val edges = edgesFile.flatMap { line =>
      val links = line.split("\\W+")
      val from = links(0)
      val to = links.tail
      for (link <- to) yield (from, link)
    }.map(e => Edge(e._1.toLong, e._2.toLong, 1))


    val verticesFile = sc.textFile("./src/main/resources/nodes_small.txt",20)

    println(edges.first())
    println(edges.collect().foreach(println))

    val vertices = verticesFile.zipWithIndex.map(_.swap)

    println(vertices.collect().foreach(println))

    val graph = Graph(vertices,edges)

    println("Triplets:")
    graph.triplets.collect.foreach(println)

    val ranks = graph.pageRank(0.001).vertices

    val swappedRanks = ranks.map(_.swap)

    val sortedRanks = swappedRanks.sortByKey(false)

    val highest = sortedRanks.first

    log.info("highest:"+highest)

    //val join = sortedRanks.join(vertices)

    //val result = join.map ( v => (v._2._1, (v._1,v._2._2))).sortByKey(false)

    log.info("Triplets:{}"+highest);

  }
}
