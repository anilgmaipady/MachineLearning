package org.ca.learn

import org.apache.spark.sql.SparkSession


object Correlation {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("Correlation example")
      .getOrCreate()
    val houses = spark.createDataFrame(Seq(
      (1620000d, 2100),
      (1690000d, 2300),
      (1400000d, 2046),
      (2000000d, 4314),
      (1060000d, 1244),
      (3830000d, 4608),
      (1230000d, 2173),
      (2400000d, 2750),
      (3380000d, 4010),
      (1480000d, 1959)
    )).toDF("price", "size");
    houses.stat.corr("price", "size");
    houses.collect().foreach(println)
  }
}



