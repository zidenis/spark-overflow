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
    val demo = false
    
    // Loading Models
//    val vectorizer = CountVectorizerModel.load(params.resourcesDir+"/vectorizer-"+id+"-"+params.qtyLDATopics)
    val ldaModel = DistributedLDAModel.load(spark.sparkContext, params.resourcesDir+"/ldaModel-"+id+"-"+params.qtyLDATopics)
    
    // Removing unused data in corpus
    var leanCorpus = corpus.drop("creationDate", "title", "document", "tags")
    if (demo) {
      println(s"Corpus Sample")
      leanCorpus.show()
    }
    // Corpus Stats
    val leanCorpusStats = leanCorpus
      .drop("id")
      .withColumn("Dataset", lit("Spark"))
      .withColumn("Questions", lit(1))
      .withColumnRenamed("answerCount", "Answers")
      .withColumnRenamed("commentCount", "Comments")
      .withColumnRenamed("viewCount", "Views")
      .withColumnRenamed("favoriteCount", "Favorites")
      .withColumnRenamed("score", "Scores")
      .select("Dataset","Questions", "Answers", "Comments", "Views", "Favorites", "Scores")
      .groupBy("Dataset")
      .sum()
    if (demo) {
      println(s"Corpus' Stats")
      leanCorpusStats.show()
    }
//    println(s"Corpus Size = ${corpus.count()}")
//    println(s"Model Size = ${ldaModel.topicDistributions.count()}")

    // Listing the topics assignments for each document in Model    
    // USING RDDs
    
//    val topicDistributions: Map[Long,Vector] = ldaModel.topicDistributions.collect().toMap
//    topicDistributions.foreach({ 
//      case (id, vect) => {
//       print(s"\n$id,")
//       vect.foreachActive({
//         // The topic is counted only if it has proportions greater than 0.1 in topics distribution
//         case (index, value) => if (value > 0.1) print("1,") else print("0,")
//       })
//      } 
//    })
//    println("")
    
    // Listing the topics assignments for each document in Model
    // USING DATAFRAMES
    
    // Creating Dataframe from LDA Model
    val topicDistributions = spark.createDataFrame(ldaModel.topicDistributions).toDF("id", "vals")
    topicDistributions.cache()
    
    val valsVector2ValsArray = udf((v: Vector) => v.toArray)
    
    val topicMembershipMatrix = topicDistributions
      .withColumn("vals", valsVector2ValsArray(col("vals")))
      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
    if (demo) {
      println(s"Topics Membership Matrix")
      topicMembershipMatrix.show()
    }
    
    val valsVector2MaxValsArray = udf((v: Vector) => {
      val maxValue = v.toArray.max
      v.toArray.map(value =>
        if (value != maxValue) 0
        else value
        )
      })
    
    val dominantTopicMatrix = topicDistributions
      .withColumn("vals", valsVector2MaxValsArray(col("vals")))
      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
    if (demo) {
      println(s"Dominant Topic Matrix")
      dominantTopicMatrix.show()
    }
    
    val valsVector2DominantTopicIndexArray = udf((v: Vector) => {
      val maxValue = v.toArray.max
      v.toArray.map(value =>
        if (value != maxValue) 0
        else 1
        )
      })
    
    val dominantTopicIndexMatrix = topicDistributions
      .withColumn("vals", valsVector2DominantTopicIndexArray(col("vals")))
      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
    if (demo) {
      println(s"Dominant Topic Matrix")
      dominantTopicIndexMatrix.show()
    }
    
    // NNDT Metric
    val nddt = dominantTopicIndexMatrix
      .drop(col("id"))
      .withColumn("Agg", lit(1))
      .groupBy("Agg")
      .sum()
      .drop(col("Agg"))
    println(s"Metric: Number of Documents with Dominant Topic K. NDDT(k)")  
    nddt.show()
    
    // Dominant Topic Proportion Metric
    val nndtColumns = nddt.columns.toSeq
    val pddt = nddt
      .select((0 until params.qtyLDATopics).map(i => (col(nndtColumns(i))/col("sum(Agg)")).alias(s"D(T${i+1})")): _*)
    println(s"Metric: Proportion of Documents with Dominant Topic K")  
    pddt.show()
    
    val threshold = 0.1
    val valsVector2ValsAboveThresholdArray = udf((v: Vector) => {
      val maxValue = v.toArray.max
      v.toArray.map(value =>
        if (value >= threshold) value
        else 0
        )
      })
    
    val topicMembershipAboveThresholdMatrix = topicDistributions
      .withColumn("vals", valsVector2ValsAboveThresholdArray(col("vals")))
      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
    if (demo) {
      println(s"Topics Membership above threshold Matrix. Threshold = ${threshold}")
      topicMembershipAboveThresholdMatrix.show()
    }
    
    // Topic Impact Metric
    val topicImpact = topicMembershipAboveThresholdMatrix
      .drop(col("id"))
      .withColumn("Agg", lit(1))
      .groupBy("Agg")
      .sum()
      .drop(col("Agg"))
    println(s"Metric: Topic Impact. Threshold = ${threshold}")  
    topicImpact.show()
    
    // Topic Share Metric
    val topicImpactColumns = topicImpact.columns.toSeq
    val topicShare = topicImpact
      .select((0 until params.qtyLDATopics).map(i => (col(topicImpactColumns(i))/col("sum(Agg)")).alias(s"Share(T${i+1})")): _*)
    println(s"Metric: Topic Share. Threshold = ${threshold}")  
    topicShare.show()
    
    // UDF to transforming the vector of document's probabilities into a vector of document-topic assignment  
    val probVectorToDocAsignVector = udf((v: Vector) => v.toArray.map(x =>
      // The topic is counted only if it has proportions greater than 0.1 in topics distribution
      if (x > 0.1) 1 
      else 0
      )
    )
   
    // UDF to transforming the vector of document's probabilities into a vector of document views
    val valsVector2ValsTimesViewCountArray = udf((viewCount: Int, v: Vector) => v.toArray.map(value => 
      if (value >= threshold) {
        value * viewCount
      } else 0
      )
    )
    
    val viewCountTopicShareMatrix = topicDistributions
      .join(leanCorpus, "id")
      .withColumn("viewCount", when(col("viewCount").isNotNull, col("viewCount")).otherwise(lit(0)))
      .withColumn("vals", valsVector2ValsTimesViewCountArray(col("viewCount"), col("vals")))
      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
    if (demo) {
      println(s"View-Count Topic Share Matrix")  
      viewCountTopicShareMatrix.show()
    }

    val totalViews = leanCorpusStats.select(col("sum(Views)")).first().getAs[Long]("sum(Views)")
    val viewCountTopicShareSum = viewCountTopicShareMatrix
      .drop(col("id"))
      .withColumn("Agg", lit(1))
      .groupBy("Agg")
      .sum()
      .drop(col("Agg"))
      .withColumn("sum(Views)", lit(totalViews))
    if (demo) {
      println(s"View-Count Topic Share Sums")  
      viewCountTopicShareSum.show()
    }
    
    // View-Count Topic Share Metric
    val viewCountTopicShareSumColumns = viewCountTopicShareSum.columns.toSeq
    val viewCountTopicShare = viewCountTopicShareSum
      .select((0 until params.qtyLDATopics).map(i => (col(viewCountTopicShareSumColumns(i))/col("sum(Views)")).alias(s"ViewShare(T${i+1})")): _*)
    println(s"Metric: View-Count Topic Share")  
    viewCountTopicShare.show()
    
    val valsVector2topicEntropyValsArray = udf((v: Vector) => v.toArray.map(value =>
      if (value >= threshold) {
        value * scala.math.log10(value)
      } else 0
    ))
    
    val topicEntropyMatrix = topicDistributions
      .withColumn("vals", valsVector2topicEntropyValsArray(col("vals")))
      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
    if (demo) {
      println(s"Topics Entropy Matrix")
      topicEntropyMatrix.show()
    }
    // Topic Entropy Metric
    val topicEntropy = topicEntropyMatrix
      .drop(col("id"))
      .withColumn("Agg", lit(1))
      .groupBy("Agg")
      .sum()
      .drop(col("Agg"))
    println(s"Metric: Topic Entropy.")  
    val topicEntropyColumns = topicEntropy.columns.toSeq
    topicEntropy
      .select((0 until params.qtyLDATopics).map(i => (col(topicEntropyColumns(i))*(-1)).alias(s"TE(T${i+1})")): _*)
      .show
      
//    // UDF to transforming the vector of document's probabilities into a vector of document scores
//    val computeDocScore = udf((score: Int, viewCount: Int, answerCount: Int, commentCount: Int, favoriteCount: Int, v: Vector) => v.toArray.map(x => 
//      if (x > 0.1) {
//        3*score + 10*commentCount + answerCount + favoriteCount
//      } else 0
//      )
//    )
//    
//    val docsScorePerTopicMatrix = topicDistributions
//      .join(leanCorpus, "id")
//      .withColumn("score", when(col("score").isNotNull, col("score")).otherwise(lit(0)))
//      .withColumn("viewCount", when(col("viewCount").isNotNull, col("viewCount")).otherwise(lit(0)))
//      .withColumn("answerCount", when(col("answerCount").isNotNull, col("answerCount")).otherwise(lit(0)))
//      .withColumn("commentCount", when(col("commentCount").isNotNull, col("commentCount")).otherwise(lit(0)))
//      .withColumn("favoriteCount", when(col("favoriteCount").isNotNull, col("favoriteCount")).otherwise(lit(0)))
//      .withColumn("vals", computeDocScore(col("score"), col("viewCount"), col("answerCount"), col("commentCount"), col("favoriteCount"), col("vals")))
//      .select(col("id") +: (0 until params.qtyLDATopics).map(i => col("vals")(i).alias(s"Topic ${i+1}")): _*)
////    docsScorePerTopicMatrix.show()
//
//    val docsScorePerTopic = docsScorePerTopicMatrix
//      .drop(col("id"))
//      .withColumn("Agg", lit(1))
//      .groupBy("Agg")
//      .sum()
//      .drop(col("Agg"))
////    docsScorePerTopic.show()
  }  

}
