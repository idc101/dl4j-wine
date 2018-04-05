import Dependencies._

lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.example",
      scalaVersion := "2.11.6",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "dl4j",
    libraryDependencies += dl4j,
    libraryDependencies += dl4jUi,
    libraryDependencies += dl4jSpark,
    libraryDependencies += dl4jSparkMl,
    libraryDependencies += nd4j,
    libraryDependencies += spark,
    libraryDependencies += sparkSql,
    libraryDependencies += sparkMl,
    libraryDependencies += hadoop,
    libraryDependencies += scalaTest % Test
  )
