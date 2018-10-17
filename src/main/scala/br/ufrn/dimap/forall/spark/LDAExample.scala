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
import org.apache.spark.mllib.clustering.OnlineLDAOptimizer
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

object LDAExample {
  
  case class Params(
    val appName       : String = "LDA"
  , val master        : String = "local[*]"
  , val resSOPosts    : String = "./resources/Posts-Spark-100.xml" // Stack Overflow's Posts Dataset
  , val resCorpusQ    : String = "./resources/CorpusQ"             // Corpus of Questions (documents are question's title and body) 
  , val resCorpusQT   : String = "./resources/CorpusQT"            // Corpus of Questions (documents are only question's title)
  , val resCorpusA    : String = "./resources/CorpusA"             // Corpus of Answers (documents are answers' body)
  , val resCorpusQA   : String = "./resources/CorpusQA"            // Corpus of Questions and answers (documents are question's title and body concatenated with all the bodies of the answers to the question) 
  , val resStopwords  : String = "./resources/stopwords.txt"       // Set of words to be ignored as tokens
  , val checkpointDir : String = "./resources/checkpoint"
//  , val master        : String = "spark://10.7.40.42:7077"
//  , val resSOPosts    : String = "hdfs://master:54310/user/hduser/stackoverflow/Posts.xml"
//  , val resCorpusQ    : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQ"
//  , val resCorpusQT   : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQT"
//  , val resCorpusA    : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusA"
//  , val resCorpusQA   : String = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQA"
//  , val resStopwords  : String = "hdfs://master:54310/user/hduser/stackoverflow/stopwords.txt"
//  , val checkpointDir : String = "hdfs://master:54310/user/hduser/stackoverflow/checkpoint"
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
  , val topDocPerTopic: Int = 5 // how many top documents per topic should be printed on output
  , val prtTopTerms   : Boolean = false
  , val prtStats      : Boolean = true
  , val describeTopics: Boolean = true
  )
  
  case class Stats(
    var LDAInitTime : Instant = Instant.now()
  , var LDAEndTime : Instant  = Instant.now()
  , var corpusSize : Long = 0
  , var vocabLenght : Int = 0
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
//    reading(spark, params)
    spark.stop()
  }
    
  case class Post(
    id:               Int,
    postTypeId:       Int,
    acceptedAnswerId: Option[Int],
    parentId:         Option[Int],
    creationDate:     Timestamp,
    score:            Int,
    viewCount:        Option[Int],
    body:             String,
    title:            Option[String],
    tags:             Option[String],
    answerCount:      Option[Int],
    favoriteCount:    Option[Int]
  )
  
  case class Document( id: Int, document: String)

  def parseXml(line: String) = {
    try {
      val xml = scala.xml.XML.loadString(line)
      val id = (xml \@ "Id").toInt
      val postTypeId = (xml \@ "PostTypeId").toInt
      val creationDate = Timestamp.valueOf(LocalDateTime.parse(xml \@ "CreationDate"))
      val score = (xml \@ "Score").toInt
      val body = (xml \@ "Body")
      var title: Option[String] = None
      var acceptedAnswerId: Option[Int] = None
      var parentId: Option[Int] = None
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
    corpusQT
      .withColumn("doc", expr("concat_ws(' ', document)"))
      .selectExpr("id", "doc")
      .write.mode(SaveMode.Overwrite).csv(params.resCorpusQT + ".csv")
       
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
    corpusQ
      .withColumn("doc", expr("concat_ws(' ', document)"))
      .selectExpr("id", "doc")
      .write.mode(SaveMode.Overwrite).csv(params.resCorpusQ + ".csv")
    
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
    corpusA
      .withColumn("doc", expr("concat_ws(' ', document)"))
      .selectExpr("id", "doc")
      .write.mode(SaveMode.Overwrite).csv(params.resCorpusA + ".csv")
    
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
    corpusQA
      .withColumn("doc", expr("concat_ws(' ', document)"))
      .selectExpr("id", "doc")
      .write.mode(SaveMode.Overwrite).csv(params.resCorpusQA + ".csv")
  }
  
  def reading(spark: SparkSession, params: Params) {
    val corpusQ = spark.read.parquet(params.resCorpusQ)
    println("\nAnalyzing Questions")
    lda_runner(corpusQ, spark, params)
//    println("Questions = " + corpusQ.count())
    val corpusA = spark.read.parquet(params.resCorpusA)
    println("\nAnalyzing Answers")
    lda_runner(corpusA, spark, params)
//    println("Answers = " + corpusA.count())
    val corpusQA = spark.read.parquet(params.resCorpusQA)
    println("\nAnalyzing Questions + Answers")
    lda_runner(corpusQA, spark, params)
//    println("Q n A = " + corpusQA.count())
    
  }
  
  def lda_runner(corpus : DataFrame, spark: SparkSession, params: Params) {
    // Tokenization
    val tokenizer = new RegexTokenizer()
      .setPattern("[\\W_]+")
      .setMinTokenLength(params.minTermLenght) // Filter away tokens with length < minTokenLenght
      .setInputCol("document")
      .setOutputCol("tokens")
    val tokenized_df = tokenizer.transform(corpus)
//    tokenized_df.select("tokens").show(false)
    
    // Removing stopwords
    val stopwords = spark.sparkContext.textFile(params.resStopwords).collect()
    var remover = new StopWordsRemover()
      .setStopWords(stopwords)
      .setInputCol("tokens")
      .setOutputCol("filtered")
    val filtered_df = remover.transform(tokenized_df)
    filtered_df.persist(MEMORY_ONLY)
//    filtered_df.select("filtered").show(false)
    
    // Top Terms
    if (params.prtTopTerms) {
      // Computing tokens frequencies
      val filtered_df_combined = filtered_df
        .select(org.apache.spark.sql.functions.explode(col("filtered")).alias("filtered_aux"))
        .select(org.apache.spark.sql.functions.collect_list("filtered_aux").alias("filtered")) 
      var countVectorizer = new CountVectorizer()
        .setInputCol("filtered")
        .setOutputCol("features")
        .fit(filtered_df_combined)
      val countVectors = countVectorizer.transform(filtered_df_combined).select("features")
      val frequency = countVectors.rdd.map(_.getAs[SparseVector]("features")).collect()(0)

      val tokensFrequency = countVectorizer.vocabulary.zip(frequency.toArray)
      println(s"Top ${params.qtyOfTopTerms} tokens:")
      tokensFrequency.take(params.qtyOfTopTerms).foreach(println)  
    }
    val stats = Stats()
    stats.corpusSize = corpus.count()
    // Runs one LDA experiment varying the number of desired Topics 
    Range.inclusive(params.minQtyLDATop, params.qtyLDATopics, 5).foreach { 
      case i => {
        params.qtyLDATopics = i
        stats.LDAInitTime = Instant.now()
        lda(filtered_df, corpus, spark, params, stats)
        stats.LDAEndTime = Instant.now()
        // Printing Stats
        if (params.prtStats) {
          println("Corpus size = " + stats.corpusSize)
          println("Vocabulary size = " + stats.vocabLenght)
          println("LDA Topics = " + params.qtyLDATopics)
          println("LDA Alpha = " + params.alpha)
          println("LDA Beta = " + params.beta)
          println("Iteractions = " + params.maxIterations)
          println("Duration = " + Duration.between(stats.LDAInitTime, stats.LDAEndTime).getSeconds + "s")
        }
      }
    }
  }
  
  def lda(filtered_df: DataFrame, corpus : DataFrame, spark: SparkSession, params: Params, stats: Stats) {
    // Computing tokens frequencies for LDA
    var vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setMinDF(params.termMinDocFreq)
      .fit(filtered_df)
    val new_countVectors = vectorizer.transform(filtered_df).select("id", "features")
    stats.vocabLenght = vectorizer.vocabulary.length
    val countVectorsMLib = MLUtils.convertVectorColumnsFromML(new_countVectors, "features")
    import spark.implicits._
    val lda_countVector = countVectorsMLib.map { case Row(id: Int, countVector: Vector) => (id.toLong, countVector) }
    
    // LDA
    val lda = new LDA()
      //.setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
      .setOptimizer(params.optimizer)
      .setK(params.qtyLDATopics)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.alpha) 
      .setTopicConcentration(params.beta)
      .setCheckpointInterval(50)
    val lda_countVectorRDD = lda_countVector.rdd
    val ldaModel = lda.run(lda_countVectorRDD)
    // DecribeTopics
    if (params.describeTopics) {
      var topicsArray = ldaModel.describeTopics(maxTermsPerTopic = params.termsPerTopic)
      var vocabList = vectorizer.vocabulary
      var topics = topicsArray.map {
        case (term, termWeight) =>
          term.map(vocabList(_)).zip(termWeight)
        }
      // Output
      topics.zipWithIndex.foreach {
        case (topic, i) =>
          println(s"TOPIC ${i+1}")
          println("---------")
          topic.foreach { 
            case (term, weight) => {
              print(s"$term (")
              println(f"$weight%2.3f) ") 
            }
          }
          if (params.topDocPerTopic > 0 && params.optimizer.equals("em")) {
            val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
            var topDocs = distLDAModel.topDocumentsPerTopic(params.topDocPerTopic)
            println("\n------")
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
