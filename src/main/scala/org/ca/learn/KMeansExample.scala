package org.ca.learn


import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession

object KMeansExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("KMeansExample").getOrCreate()
    val data = spark.createDataFrame(Seq(
      Vectors.dense(12839,2405),
      Vectors.dense(10000,2200),
      Vectors.dense(8040,1400),
      Vectors.dense(13104,1800),
      Vectors.dense(10000,2351),
      Vectors.dense(3049,795),
      Vectors.dense(38768,2725),
      Vectors.dense(16250,2150),
      Vectors.dense(43026,2724),
      Vectors.dense(44431,2675),
      Vectors.dense(40000,2930),
      Vectors.dense(1260,870),
      Vectors.dense(15000,2210),
      Vectors.dense(10032,1145),
      Vectors.dense(12420,2419),
      Vectors.dense(69696,2750),
      Vectors.dense(12600,2035),
      Vectors.dense(10240,1150),
      Vectors.dense(876,665),
      Vectors.dense(8125,1430),
      Vectors.dense(11792,1920),
      Vectors.dense(1512,1230),
      Vectors.dense(1276,975),
      Vectors.dense(67518,2400),
      Vectors.dense(9810,1725),
      Vectors.dense(6324,2300),
      Vectors.dense(12510,1700),
      Vectors.dense(15616,1915),
      Vectors.dense(15476,2278),
      Vectors.dense(13390,2497.5),
      Vectors.dense(1158,725),
      Vectors.dense(2000,870),
      Vectors.dense(2614,730),
      Vectors.dense(13433,2050),
      Vectors.dense(12500,3330),
      Vectors.dense(15750,1120),
      Vectors.dense(13996,4100),
      Vectors.dense(10450,1655),
      Vectors.dense(7500,1550),
      Vectors.dense(12125,2100),
      Vectors.dense(14500,2100),
      Vectors.dense(10000,1175),
      Vectors.dense(10019,2047.5),
      Vectors.dense(48787,3998),
      Vectors.dense(53579,2688),
      Vectors.dense(10788,2251),
      Vectors.dense(11865,1906)
    ).map(Tuple1.apply)).toDF("features")

    val kmeans = new KMeans().setK(4).setSeed(1L)
    val model = kmeans.fit(data)

    //belong to cluster 3
    val prediction1 = model.transform(spark.createDataFrame(Seq(Vectors.dense(876,665)).map(Tuple1.apply)).toDF("features"))
    prediction1.show(20,false)

    //belong to cluster 0
    val prediction2 = model.transform(spark.createDataFrame(Seq(Vectors.dense(15750,1120)).map(Tuple1.apply)).toDF("features"))
    prediction2.show(20,false)


  }
}
