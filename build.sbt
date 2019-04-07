name := "SparkOverflow"

version := "1.0"

organization := "br.ufrn.dimap.forall.spark"

scalaVersion := "2.11.11"

val sparkVersion = "2.2.0"

resolvers ++= Seq("spark-stemming" at "https://dl.bintray.com/spark-packages/maven/")

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-streaming" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-hive" % sparkVersion % "provided",
  "com.databricks" %% "spark-xml" % "0.4.1",
  "master" % "spark-stemming" % "0.2.0",
  "com.typesafe" % "config" % "1.3.2"
)
