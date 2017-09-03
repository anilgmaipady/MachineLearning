package org.ca.learn
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
object TfIdf {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("TfIdf")
      .getOrCreate()
    val sentenceData = spark.read.option("delimiter", "\n").csv("./src/main/resources/hardtimes/").toDF("sentence")

    //sentenceData.collect().foreach(println)

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")

    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)


    val featurizedData = hashingTF.transform(wordsData)
    featurizedData.show(10,false)

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)

    rescaledData.show(10,false)


  }

}
