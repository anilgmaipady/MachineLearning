name := "examples"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.0"


libraryDependencies ++= Seq(
  //"com.amazonaws" % "aws-java-sdk" % "1.11.185",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "org.apache.hadoop" % "hadoop-common" % "2.8.1",
  "org.apache.hadoop" % "hadoop-aws" % "2.8.1"
)


