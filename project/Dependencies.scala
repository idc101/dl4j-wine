import sbt._

object Dependencies {
  lazy val dl4j = "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1"
  lazy val dl4jUi = "org.deeplearning4j" %% "deeplearning4j-ui" % "0.9.1"
  lazy val dl4jSpark = "org.deeplearning4j" %% "dl4j-spark" % "0.9.1_spark_2"
  lazy val dl4jSparkMl = "org.deeplearning4j" %% "dl4j-spark-ml" % "0.9.1_spark_2"
  lazy val nd4j = "org.nd4j" % "nd4j-native-platform" % "0.9.1"
  lazy val spark = "org.apache.spark" %% "spark-core" % "2.2.0"
  lazy val sparkSql = "org.apache.spark" %% "spark-sql" % "2.2.0"
  lazy val sparkMl = "org.apache.spark" %% "spark-mllib" % "2.2.0"
  lazy val hadoop = "org.apache.hadoop" % "hadoop-client" % "2.7.2"
  lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.0.5"
}
