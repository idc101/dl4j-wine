import sbt._

object Dependencies {
  lazy val dl4j = "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1"
  lazy val dl4jUi = "org.deeplearning4j" %% "deeplearning4j-ui" % "0.9.1"
  lazy val nd4j = "org.nd4j" % "nd4j-native-platform" % "0.9.1"
  lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.0.5"
}
