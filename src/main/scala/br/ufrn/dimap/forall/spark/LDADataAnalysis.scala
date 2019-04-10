package br.ufrn.dimap.forall.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.CountVectorizerModel
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel._
import java.time.LocalDateTime
import java.time.Instant
import java.time.Duration
import java.sql.Timestamp
import org.apache.spark.SparkConf
import org.apache.spark.mllib.feature.Stemmer
import scala.collection.mutable.ListBuffer
import scala.beans.BeanProperty
import com.typesafe.config.ConfigFactory
import com.typesafe.config.ConfigBeanFactory

object LDADataAnalysis {
  
  case class Params(
    @BeanProperty var appName       : String

// Cluster Configurations
  , @BeanProperty var master        : String
  , @BeanProperty var resSOPosts    : String
  , @BeanProperty var resCorpusQ    : String
  , @BeanProperty var resCorpusQT   : String
  , @BeanProperty var resCorpusQA   : String
  , @BeanProperty var resStopwords  : String
  , @BeanProperty var checkpointDir : String
  , @BeanProperty var resourcesDir : String

// Experiment Configurations
  , @BeanProperty var minTermLenght : Int
  , @BeanProperty var qtyOfTopTerms : Int
  , @BeanProperty var termMinDocFreq: Int
  , @BeanProperty var qtyLDATopics  : Int
  , @BeanProperty var minQtyLDATop  : Int
  , @BeanProperty var optimizer     : String
  , @BeanProperty var alpha         : Double
  , @BeanProperty var beta          : Double
  , @BeanProperty var maxIterations : Int
  , @BeanProperty var termsPerTopic : Int
  , @BeanProperty var topDocPerTopic: Int
  , @BeanProperty var prtTopTerms   : Boolean
  , @BeanProperty var prtStats      : Boolean
  , @BeanProperty var describeTopics: Boolean
  , @BeanProperty var horizontOutput: Boolean
  , @BeanProperty var corpusQT      : Boolean
  , @BeanProperty var corpusQ       : Boolean
  , @BeanProperty var corpusQA      : Boolean
  , @BeanProperty var outputCSV     : Boolean
  ) {
  def this() = this("","","","","","","","","",0,0,0,0,0,"", 0,0,0,0,0,false,false,false,false,false,false,false,false)
}
  
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR) // Set the log level to only print errors
    
    val config = ConfigFactory.parseFile(new java.io.File("application.conf")).getConfig("config")
    val params = ConfigBeanFactory.create(config, classOf[Params])
    
    println(s"Experiment ${params}")
    
    val spark = SparkSession
      .builder
      .appName(params.appName)
      .master(params.master)
      .getOrCreate()
    spark.sparkContext.setCheckpointDir(params.checkpointDir)
    
    analysis(spark, params)
    
    
    spark.stop()
  }
  
  def analysis(spark: SparkSession, params: Params) {
    if (params.corpusQT) {
      val corpusQT = spark.read.parquet(params.resCorpusQT + ".parquet")
      Range.inclusive(params.minQtyLDATop, params.qtyLDATopics, 10).foreach { 
        case i => {
          params.qtyLDATopics = i
          lda_analysis("QT", corpusQT, spark, params)
        }
      }
    }
    if (params.corpusQ) {
      val corpusQ = spark.read.parquet(params.resCorpusQ + ".parquet")
      Range.inclusive(params.minQtyLDATop, params.qtyLDATopics, 10).foreach { 
        case i => {
          params.qtyLDATopics = i
          lda_analysis("Q", corpusQ, spark, params)
        }
      }
    }
    if (params.corpusQA) {
      val corpusQA = spark.read.parquet(params.resCorpusQA + ".parquet")
      Range.inclusive(params.minQtyLDATop, params.qtyLDATopics, 10).foreach { 
        case i => {
          params.qtyLDATopics = i
          lda_analysis("QA", corpusQA, spark, params)
        }
      }
    }
  }
  
  def lda_analysis(id : String, corpus: DataFrame, spark: SparkSession, params: Params) {
    
    // Loading Models
    val vectorizer = CountVectorizerModel.load(params.resourcesDir+"/vectorizer-"+id+"-"+params.qtyLDATopics)
    val ldaModel = DistributedLDAModel.load(spark.sparkContext, params.resourcesDir+"/ldaModel-"+id+"-"+params.qtyLDATopics)
    
//    println(s"Corpus Size = ${corpus.count()}")
//    println(s"Model Size = ${ldaModel.topicDistributions.count()}")

    // Listing the topics assignments for each document in Model
    // The topic is counted only if it has proportions greater than 0.1 in topics distribution
    
    // Listing using RDDs
    val topicDistributions: Map[Long,Vector] = ldaModel.topicDistributions.collect().toMap
    topicDistributions.foreach({ 
      case (id, vect) => {
       print(s"\n$id,")
       vect.foreachActive({
         case (index, value) => if (value > 0.1) print("1,") else print("0,")
       })
      } 
    })
    println("")
    
    // Listing using Dataframes
    val vecToSeq = udf((v: Vector) => v.toArray.map(x => if (x > 0.1) 1 else 0))
    val topicDistributionsDF = spark.createDataFrame(ldaModel.topicDistributions).toDF("id", "vals")
      .withColumn("vals", vecToSeq(col("vals")))
      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
    topicDistributionsDF.show()
    
  }
}