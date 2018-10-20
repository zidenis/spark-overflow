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

object LDAExample {
  
  case class Params(
    val appName       : String = "LDA"
//  , val master        : String = "local[*]"
//  , val resSOPosts    : String = "./resources/Posts-Spark-100.xml" // Stack Overflow's Posts Dataset
//  , val resCorpusQ    : String = "./resources/CorpusQ"             // Corpus of Questions (documents are question's title and body) 
//  , val resCorpusQT   : String = "./resources/CorpusQT"            // Corpus of Questions (documents are only question's title)
//  , val resCorpusA    : String = "./resources/CorpusA"             // Corpus of Answers (documents are answers' body)
//  , val resCorpusQA   : String = "./resources/CorpusQA"            // Corpus of Questions and answers (documents are question's title and body concatenated with all the bodies of the answers to the question) 
//  , val resStopwords  : String = "./resources/stopwords.txt"       // Set of words to be ignored as tokens
//  , val checkpointDir : String = "./resources/checkpoint"
  , val master        : String = "spark://10.7.40.42:7077"
  , val resSOPosts    : String = "hdfs://master:54310/user/hduser/stackoverflow/Posts.xml"
  , val resCorpusQ    : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQ"
  , val resCorpusQT   : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQT"
  , val resCorpusA    : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusA"
  , val resCorpusQA   : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQA"
  , val resStopwords  : String = "hdfs://master:54310/user/hduser/stackoverflow/stopwords.txt"
  , val checkpointDir : String = "hdfs://master:54310/user/hduser/stackoverflow/checkpoint"
  , val minTermLenght : Int = 3  // A term should have at least minTermLenght characters to be considered as token
  , val qtyOfTopTerms : Int = 20 // how many top terms should be printed on output
  , val termMinDocFreq: Int = 3  // minimum number of different documents a term must appear in to be included in the vocabulary
  , var qtyLDATopics  : Int = 40 // number of LDA latent topics
  , val minQtyLDATop  : Int = 20
  , val optimizer     : String = "em"
  , val alpha         : Double = -1 // LDA dirichlet prior probability placed on document-topic distribution. Choose a low alpha if your documents are made up of a few dominant topics 
  , val beta          : Double = -1 // LDA dirichlet prior probability placed on topic-word distribution. Choose a low beta if your topics are made up of a few dominant words
  , val maxIterations : Int = 1000 // number of LDA training iterations
  , val termsPerTopic : Int = 20 // how many terms per topic should be printed on output 
  , val topDocPerTopic: Int = 10 // how many top documents per topic should be printed on output
  , val prtTopTerms   : Boolean = true
  , val prtStats      : Boolean = true
  , val describeTopics: Boolean = true
  , val horizontOutput: Boolean = true
  )
  
  case class Stats(
    var LDAInitTime : Instant = Instant.now()
  , var LDAEndTime : Instant  = Instant.now()
  , var corpusSize : Long = 0
  , var vocabLength : Int = 0
  , var alpha : Double = 0
  , var beta : Double = 0
  )
  
  case class Post(
    id:               Long,
    postTypeId:       Int,
    acceptedAnswerId: Option[Long],
    parentId:         Option[Long],
    creationDate:     Timestamp,
    score:            Int,
    viewCount:        Option[Int],
    body:             String,
    title:            Option[String],
    tags:             Option[String],
    answerCount:      Option[Int],
    favoriteCount:    Option[Int]
  )
  
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR) // Set the log level to only print errors
    
    val params = Params()
    
    val spark = SparkSession
      .builder
      .appName(params.appName)
      .master(params.master)
      .getOrCreate()
    spark.sparkContext.setCheckpointDir(params.checkpointDir)
    
    processing(spark, params)
    reading(spark, params)
    spark.stop()
  }
    
  def parseXml(line: String) = {
    try {
      val xml = scala.xml.XML.loadString(line)
      val id = (xml \@ "Id").toLong
      val postTypeId = (xml \@ "PostTypeId").toInt
      val creationDate = Timestamp.valueOf(LocalDateTime.parse(xml \@ "CreationDate"))
      val score = (xml \@ "Score").toInt
      val body = (xml \@ "Body")
      var title: Option[String] = None
      var acceptedAnswerId: Option[Long] = None
      var parentId: Option[Long] = None
      var tags: Option[String] = None
      var viewCount: Option[Int] = None
      var answerCount: Option[Int] = None
      var favoriteCount: Option[Int] = None
      if (postTypeId == 1) {
        title = Some(xml \@ "Title")
        tags = Some(xml \@ "Tags")
        var temp = (xml \@ "AcceptedAnswerId")
        acceptedAnswerId = if (temp.isEmpty()) None else Some(temp.toInt)
        temp = (xml \@ "ViewCount")
        viewCount = if (temp.isEmpty()) None else Some(temp.toInt)
        temp = (xml \@ "AnswerCount")
        answerCount = if (temp.isEmpty()) None else Some(temp.toInt)
        temp = (xml \@ "FavoriteCount")
        favoriteCount = if (temp.isEmpty()) None else Some(temp.toInt)
      }
      if (postTypeId == 2) {
        var temp = (xml \@ "ParentId")
        parentId = if (temp.isEmpty()) None else Some(temp.toInt)
      }
      Some(
        Post(
          id,
          postTypeId,
          acceptedAnswerId,
          parentId,
          creationDate,
          score,
          viewCount,
          body,
          title,
          tags,
          answerCount,
          favoriteCount
        )
      )
    } catch {
      case e: Exception => None
    }
  }
  
  def isSparkRelated(tags : String) = {
    tags.contains("apache-spark") || 
    tags.contains("pyspark") ||
    tags.contains("sparklyr") ||
    tags.contains("sparkr") ||
    tags.contains("spark-dataframe") ||
    tags.contains("spark-streaming") ||
    tags.contains("spark-cassandra-connector") ||
    tags.contains("spark-graphx") ||
    tags.contains("spark-submit") ||
    tags.contains("spark-structured-streaming") ||
    tags.contains("spark-observer") ||
    tags.contains("spark-csv") ||
    tags.contains("spark-avro") ||
    tags.contains("spark-hive")
  }
  
  def cleanDocument(document : String) = {
    document
      .filter(_ >= ' ') // throw away all control characters.
      .toLowerCase() // put all chars in lowercase
      .replaceAll("<pre>.+</pre>", " ") // remove code parts
      .replaceAll("<([^>])+>", " ") // remove tags
      .replaceAll("\\d+((\\.\\d+)?)*", " ") // remove numbers like 2018, 1.2.1 
      .replaceAll("[!@#$%^&*()_=+.,<>:;?/{}`~'\"\\[\\]\\-]", " ") // remove special characters
  }

  def processing(spark: SparkSession, params: Params) {
  
    val lines = spark.sparkContext.textFile(params.resSOPosts).flatMap(parseXml)
    
    import spark.implicits._
    import spark.sql

    spark.udf.register("sparkRelated", (tags : String) =>  isSparkRelated(tags))
    spark.udf.register("cleanDocument", (document : String) =>  cleanDocument(document))
    
    // Obter posts para somente aqueles em que eh possivel ter pergunta sobre Apache Spark
    val posts = lines
      .toDS()
      .where("year(creationDate) > 2012") // Primeira pergunta sobre spark eh de 2013
      
    // Obter Posts com perguntas sobre Spark
    val sparkQuestions = posts
      .where("postTypeId = 1")  // somente perguntas
      .withColumn("sparkRelated", expr("sparkRelated(tags)")) // posts com tag de spark
      .where("sparkRelated") // somente com tags de spark
    sparkQuestions.persist(MEMORY_ONLY)
    
    // Corpus QT
    // Clean document's string
    val cleanedQT = sparkQuestions
      .withColumn("cleaned", expr("cleanDocument(title)"))
      
    // Tokenization
    val tokenizer = new RegexTokenizer()
      .setPattern("[\\W_]+")
      .setMinTokenLength(params.minTermLenght) // Filter away tokens with length < minTokenLenght
      .setInputCol("cleaned")
      .setOutputCol("tokenized")
    val tokenizedQT = tokenizer.transform(cleanedQT)
    
    // Removing stopwords
    val stopwords = spark.sparkContext.textFile(params.resStopwords).collect()
    val remover = new StopWordsRemover()
      .setStopWords(stopwords)
      .setInputCol("tokenized")
      .setOutputCol("removed")
    val removedQT = remover.transform(tokenizedQT)
    
    // Stemming
    val stemmer = new Stemmer()
      .setInputCol("removed")
      .setOutputCol("document")
      .setLanguage("English")
    val stemmedQT = stemmer.transform(removedQT)
    
    // Corpus de Perguntas em que cada documento eh apenas o titulo da pergunta
    val corpusQT = stemmedQT
      .select("id", "creationDate", "score", "viewCount", "title", "document", "tags", "answerCount", "favoriteCount")
    
//    corpusQT.show()
    corpusQT.write.mode(SaveMode.Overwrite).parquet(params.resCorpusQT + ".parquet")
//  corpusQT
//    .withColumn("doc", expr("concat_ws(' ', document)"))
//    .selectExpr("id", "doc")
//    .write.mode(SaveMode.Overwrite).csv(params.resCorpusQT + ".csv")
       
    // Corpus Q
    val cleanedQ = sparkQuestions
      .withColumn("title_body", concat($"title", lit(" "), $"body"))
      .withColumn("cleaned", expr("cleanDocument(title_body)"))  
    cleanedQ.persist(MEMORY_ONLY)
    cleanedQ.createOrReplaceTempView("corpusQ")
    
    val tokenizedQ = tokenizer.transform(cleanedQ)
    val removedQ = remover.transform(tokenizedQ)
    val stemmedQ = stemmer.transform(removedQ)
    
    // Corpus de Perguntas em que cada documento eh o titulo cocatenado com o corpo da pergunta
    val corpusQ = stemmedQ
      .select("id", "creationDate", "score", "viewCount", "title", "document", "tags", "answerCount", "favoriteCount")
    
//    corpusQ.show()
    corpusQ.write.mode(SaveMode.Overwrite).parquet(params.resCorpusQ + ".parquet")
//  corpusQ
//    .withColumn("doc", expr("concat_ws(' ', document)"))
//    .selectExpr("id", "doc")
//    .write.mode(SaveMode.Overwrite).csv(params.resCorpusQ + ".csv")
    
    // Obter Posts com respostas a perguntas sobre Spark
    val stackAnswers = posts
      .where("postTypeId = 2") // somente respostas
    stackAnswers.createOrReplaceTempView("stackAnswers")
      
    // Corpus A
    // Corpus de Respostas em que cada documento eh a concatenacao dos corpos das respostas dada a cada pergunta
    val cleanedA = sql("""
      SELECT answers.id, corpusQ.title, answers.cleaned 
        FROM (
      SELECT a.parentId as id
           , cleanDocument(concat_ws(' ', collect_list(a.body))) as cleaned
        FROM stackAnswers a 
   LEFT SEMI JOIN corpusQ q 
          ON a.parentId = q.id 
    GROUP BY a.parentId
        ) as answers
   LEFT JOIN corpusQ
          ON answers.id = corpusQ.id       
    """)
    cleanedA.persist(MEMORY_ONLY)
    cleanedA.createOrReplaceTempView("corpusA")
    
    val tokenizedA = tokenizer.transform(cleanedA)
    val removedA = remover.transform(tokenizedA)
    val stemmedA = stemmer.transform(removedA)
    
    val corpusA = stemmedA
      .select("id", "title", "document")
    
//    corpusA.show()
    corpusA.write.mode(SaveMode.Overwrite)parquet(params.resCorpusA + ".parquet")
//  corpusA
//    .withColumn("doc", expr("concat_ws(' ', document)"))
//    .selectExpr("id", "doc")
//    .write.mode(SaveMode.Overwrite).csv(params.resCorpusA + ".csv")
    
    // Corpus QA
    // Obter Posts com perguntas sobre Spark e suas respectivas respostas
    val sparkQA = sql("""
      SELECT q.id, q.title, q.cleaned qd, a.cleaned ad, q.creationDate, q.score, q.viewCount, q.tags, q.answerCount, q.favoriteCount
        FROM corpusQ q 
   LEFT JOIN corpusA a 
          ON q.id = a.id
    """)
    
    val cleanedQA = sparkQA
      .withColumn("ad_not_null", coalesce($"ad",lit(""))) // para o caso de perguntas sem respostas
      .withColumn("cleaned", concat($"qd", lit(" "), $"ad_not_null"))
      
    val tokenizedQA = tokenizer.transform(cleanedQA)
    val removedQA = remover.transform(tokenizedQA)
    val stemmedQA = stemmer.transform(removedQA)  
    
    val corpusQA = stemmedQA
      .select("id", "creationDate", "score", "viewCount", "title", "document", "tags", "answerCount", "favoriteCount")

//    corpusQA.show()
    corpusQA.write.mode(SaveMode.Overwrite).parquet(params.resCorpusQA + ".parquet")
//  corpusQA
//    .withColumn("doc", expr("concat_ws(' ', document)"))
//    .selectExpr("id", "doc")
//    .write.mode(SaveMode.Overwrite).csv(params.resCorpusQA + ".csv")
  }
  
  def reading(spark: SparkSession, params: Params) {
    val corpusQT = spark.read.parquet(params.resCorpusQT + ".parquet")
    var stats = Stats()
    stats.corpusSize = corpusQT.count()
    lda_runner("QT", corpusQT, spark, params, stats)
    
    val corpusQ = spark.read.parquet(params.resCorpusQ + ".parquet")
    stats = Stats()
    stats.corpusSize = corpusQ.count()
    lda_runner("Q", corpusQ, spark, params, stats)

    val corpusA = spark.read.parquet(params.resCorpusA + ".parquet")
    stats = Stats()
    stats.corpusSize = corpusA.count()
    lda_runner("A", corpusA, spark, params, stats)

    val corpusQA = spark.read.parquet(params.resCorpusQA + ".parquet")
    stats = Stats()
    stats.corpusSize = corpusQA.count()
    lda_runner("QA", corpusQA, spark, params, stats)
  }
  
  def lda_runner(id : String, corpus : DataFrame, spark: SparkSession, params: Params, stats: Stats) {
    println(s"Analyzing Corpus $id")
    // Removing additional stopwords
    val stopwords = Array("apache", "spark")
    val remover = new StopWordsRemover()
      .setStopWords(stopwords)
      .setInputCol("document")
      .setOutputCol("removed")
    val removed = remover.transform(corpus)
    
    // Top Terms in whole corpus
    if (params.prtTopTerms) {
      removed.persist(MEMORY_ONLY)
      val combined = removed
        .select(org.apache.spark.sql.functions.explode(col("removed")).alias("exploded"))
        .select(org.apache.spark.sql.functions.collect_list("exploded").alias("combined"))
      val countVectorizer = new CountVectorizer()
        .setInputCol("combined")
        .setOutputCol("features")
        .fit(combined)
      val countVectors = countVectorizer.transform(combined).select("features")
      val frequency = countVectors.rdd.map(_.getAs[SparseVector]("features")).collect()(0)
      val tokensFrequency = countVectorizer.vocabulary.zip(frequency.toArray)
      print(s"\nTop ${params.qtyOfTopTerms} tokens: ")
      tokensFrequency.take(params.qtyOfTopTerms).foreach(tuple => print(f"${tuple._1} (${tuple._2}%.0f) "))
      println("")
    }
    
    // Runs one LDA experiment varying the number of desired Topics 
    Range.inclusive(params.minQtyLDATop, params.qtyLDATopics, 5).foreach { 
      case i => {
        params.qtyLDATopics = i
        stats.LDAInitTime = Instant.now()
        lda(removed, spark, params, stats)
        stats.LDAEndTime = Instant.now()
        // Printing Stats
        if (params.prtStats) {
          println("Corpus size = " + stats.corpusSize)
          println("Vocabulary size = " + stats.vocabLength)
          println("LDA Topics = " + params.qtyLDATopics)
          println("LDA Alpha = " + stats.alpha)
          println("LDA Beta = " + stats.beta)
          println("Iteractions = " + params.maxIterations)
          println("Duration = " + Duration.between(stats.LDAInitTime, stats.LDAEndTime).getSeconds + "s")
        }
      }
    }
  }
  
  def lda(corpus: DataFrame, spark: SparkSession, params: Params, stats: Stats) {
    // Computing tokens frequencies for LDA
    val vectorizer = new CountVectorizer()
      .setInputCol("removed")
      .setOutputCol("features")
      .setMinDF(params.termMinDocFreq)
      .setMinTF(1.0)
      .fit(corpus)
    val vectorized = vectorizer.transform(corpus).select("id", "features")
    stats.vocabLength = vectorizer.vocabulary.length
    
    // To use the LDA from MLlib library
    val vectorizedMLib = MLUtils.convertVectorColumnsFromML(vectorized, "features")
    import spark.implicits._
    val vectorizedDS = vectorizedMLib.map { case Row(id: Long, countVector: Vector) => (id, countVector) }
    val vectorizedRDD = vectorizedDS.rdd
    
    // Setting default LDA Parameters
    var docConcentration = params.alpha
    var topicConcentration = params.beta
    if (params.optimizer.equals("em")) {
      if (params.alpha == -1) docConcentration = (50/params.qtyLDATopics)+1
      if (params.beta == -1) topicConcentration = 1.1
    } else if (params.optimizer.equals("online")){
      if (params.alpha == -1) docConcentration = 1/params.qtyLDATopics
      if (params.beta == -1) topicConcentration = 1/params.qtyLDATopics
    }
    
    val lda = new LDA()
      .setOptimizer(params.optimizer)
      .setK(params.qtyLDATopics)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(docConcentration) 
      .setTopicConcentration(topicConcentration)
      .setCheckpointInterval(50)
    val ldaModel = lda.run(vectorizedRDD)    
    stats.alpha = ldaModel.docConcentration(0)
    stats.beta = ldaModel.topicConcentration

    // Describing Topics
    if (params.describeTopics) {
      val vocabList = vectorizer.vocabulary
      val describedTopics = ldaModel.describeTopics(maxTermsPerTopic = params.termsPerTopic)
      val topics = describedTopics.map {
        case (terms, weights) => {
          val zipped = terms.map(vocabList(_)).zip(weights)
          zipped.map {
            case (term, weight) => f"$term ($weight%2.3f)"
          }
        }
      }
      
      // Output with Horizontal layout using dataframes
      if (params.horizontOutput) {
        // Transforming the Array of Topics in a Dataframe with one column for each topic
        val transposed = topics.transpose
        val transposedDF = spark.sparkContext.parallelize(transposed).toDF()
        val topicsDF = transposedDF.select((Range(0, params.qtyLDATopics).map(i => $"value"(i) as "TOPIC " + (i+1)):_*))
        // Getting the top documents per topic
        if (params.topDocPerTopic > 0 && params.optimizer.equals("em")) {
          var columns = ListBuffer[ListBuffer[String]]()
          Range(0, params.qtyLDATopics).map( i => {
            var lines = ListBuffer[String]()
            val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
            val topDocs = distLDAModel.topDocumentsPerTopic(params.topDocPerTopic)
            (topDocs(i)._1).zip(topDocs(i)._2).foreach {
              case (id, weight) => {
                lines += s"$id : " + corpus.where($"id" === id).select("title").first().mkString(" : ") + f" : $weight%2.4f"
              }
            }
            columns += lines            
          })
          val questions = columns.transpose.toDF()
          val questionsDF = questions.select((Range(0, params.qtyLDATopics).map(i => $"value"(i) as "TOPIC " + (i+1)):_*))
          topicsDF.union(questionsDF).show(params.qtyLDATopics+params.qtyOfTopTerms+params.topDocPerTopic , false)
        }
      } 
      else {
        // Output with Vertical layout
        topics.zipWithIndex.foreach {
          case (topic, i) =>
            println(s"TOPIC ${i+1}")
            println("---------")
            topic.foreach(println)
            if (params.topDocPerTopic > 0 && params.optimizer.equals("em")) {
              val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
              val topDocs = distLDAModel.topDocumentsPerTopic(params.topDocPerTopic)
              println("---------")
              val temp = (topDocs(i)._1).zip(topDocs(i)._2)
              temp.foreach {
                case (id, weight) => {
                  print(f"$weight%2.4f : $id : ")
                  println(corpus.where($"id" === id).select("title").first().getAs[String]("title"))
                }
              }
            }
            println(s"\n==========")
        }
      }
    }
  }
}
