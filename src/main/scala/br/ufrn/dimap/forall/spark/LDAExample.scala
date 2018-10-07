package br.ufrn.dimap.forall.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.CountVectorizer
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
import java.sql.Timestamp

object LDAExample {
  
  val resourceInput = "./resources/Posts-Spark-100.xml"
  val corpusQoutput = "./resources/CorpusQ.parquet"
  val corpusAoutput = "./resources/CorpusA.parquet"
  val corpusQAoutput = "./resources/CorpusQA.parquet"
  val stopwordsFile = "./resources/stopwords.txt"
  
//  val resourceInput = "hdfs://master:54310/user/hduser/stackoverflow/Posts.xml"
//  val corpusQoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQ.parquet"
//  val corpusAoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusA.parquet"
//  val corpusQAoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQA.parquet"
//  val stopwordsFile = "hdfs://master:54310/user/hduser/stackoverflow/stopwords.txt"
  
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR) // Set the log level to only print errors
    val spark = SparkSession
      .builder
      .appName("LDAExample")
      .master("local[*]")
      .getOrCreate()
      
//    processing(spark)
    reading(spark)
    spark.stop()
  }
    
  case class Post(
    id:               Int,
    postTypeId:       Int,
//    acceptedAnswerId: Option[Int],
    parentId:         Option[Int],
    creationDate:     Timestamp,
//    score:            Int,
//    viewCount:        Option[Int],
    body:             String,
    title:            Option[String],
    tags:             Option[String]
//    answerCount:      Option[Int],
//    favoriteCount:    Option[Int]
  )
  
  case class Document( id: Int, document: String)

  def parseXml(line: String) = {
    try {
      val xml = scala.xml.XML.loadString(line)
      val id = (xml \@ "Id").toInt
      val postTypeId = (xml \@ "PostTypeId").toInt
      val creationDate = Timestamp.valueOf(LocalDateTime.parse(xml \@ "CreationDate"))
//      val score = (xml \@ "Score").toInt
      val body = (xml \@ "Body")
      var title: Option[String] = None
//      var acceptedAnswerId: Option[Int] = None
      var parentId: Option[Int] = None
      var tags: Option[String] = None
//      var viewCount: Option[Int] = None
//      var answerCount: Option[Int] = None
//      var favoriteCount: Option[Int] = None
      if (postTypeId == 1) {
        title = Some(xml \@ "Title")
        tags = Some(xml \@ "Tags")
//        var temp = (xml \@ "AcceptedAnswerId")
//        acceptedAnswerId = if (temp.isEmpty()) None else Some(temp.toInt)
//        temp = (xml \@ "ViewCount")
//        viewCount = if (temp.isEmpty()) None else Some(temp.toInt)
//        temp = (xml \@ "AnswerCount")
//        answerCount = if (temp.isEmpty()) None else Some(temp.toInt)
//        temp = (xml \@ "FavoriteCount")
//        favoriteCount = if (temp.isEmpty()) None else Some(temp.toInt)
      }
      if (postTypeId == 2) {
        var temp = (xml \@ "ParentId")
        parentId = if (temp.isEmpty()) None else Some(temp.toInt)
      }
      Some(
        Post(
          id,
          postTypeId,
//          acceptedAnswerId,
          parentId,
          creationDate,
//          score,
//          viewCount,
          body,
          title,
          tags
//          answerCount,
//          favoriteCount
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
  }

  def processing(spark: SparkSession) {
  
    val lines = spark.sparkContext.textFile(resourceInput).flatMap(parseXml)
    
    import spark.implicits._
    import spark.sql

    // Obter posts para somente aqueles em que eh possivel ter pergunta sobre Apache Spark
    val posts = lines
      .toDS()
      .where("year(creationDate) > 2012") // Primeira pergunta sobre spark eh de 2013
  
    // Obter Posts com perguntas sobre Spark 
    spark.udf.register("sparkRelated", (tags : String) =>  isSparkRelated(tags))
    spark.udf.register("cleanDocument", (document : String) =>  cleanDocument(document))
    val sparkQuestions = posts
      .where("postTypeId = 1")  // somente perguntas
      .withColumn("sparkRelated", expr("sparkRelated(tags)")) // posts com tag de spark
      .where("sparkRelated") // somente com tags de spark
//    sparkQuestions.show()
    val corpusQ = sparkQuestions
      .withColumn("title_body", concat($"title", lit(" "), $"body"))
      .withColumn("document", expr("cleanDocument(title_body)"))
      .select("id","title","document")
    corpusQ.persist(MEMORY_ONLY)
    corpusQ.createOrReplaceTempView("corpusQ")
//    println("Questions = " + corpusQ.count())
//    corpusQ.show(20, false)
    corpusQ.write.mode(SaveMode.Overwrite).parquet(corpusQoutput)

    // Obter Posts com respostas a perguntas sobre Spark
    val stackAnswers = posts
      .where("postTypeId = 2") // somente respostas
    stackAnswers.createOrReplaceTempView("stackAnswers")
//    val sparkAnswers = stackAnswers
//      .join(sparkQuestions, stackAnswers("parentId") === sparkQuestions("id"), "leftsemi")
    val corpusA = sql("""
      SELECT answers.id, corpusQ.title, answers.document 
        FROM (
      SELECT a.parentId as id
           , cleanDocument(concat_ws(' ', collect_list(a.body))) as document 
        FROM stackAnswers a 
   LEFT SEMI JOIN corpusQ q 
          ON a.parentId = q.id 
    GROUP BY a.parentId
        ) as answers
   LEFT JOIN corpusQ
          ON answers.id = corpusQ.id
    """)
    corpusA.persist(MEMORY_ONLY)
    corpusA.createOrReplaceTempView("corpusA")
//    println("Answers = " + corpusA.count())
//    corpusA.show(10, false)
    corpusA.write.mode(SaveMode.Overwrite)parquet(corpusAoutput)
    
    // Obter Posts com perguntas sobre Spark e suas respectivas perguntas
    val sparkQA = sql("""
      SELECT q.id, q.title, q.document qd, a.document ad 
        FROM corpusQ q 
   LEFT JOIN corpusA a 
          ON q.id = a.id
    """)
    val corpusQA = sparkQA
      .withColumn("ad_not_null", coalesce($"ad",lit("")))
      .withColumn("document", concat($"qd", lit(" "), $"ad_not_null"))
      .select("id","title","document")
//    println("QA = " + corpusQA.count())
//    corpusQA.show(10, false)
    corpusQA.write.mode(SaveMode.Overwrite).parquet(corpusQAoutput)
  }
  
  def reading(spark: SparkSession) {
    val corpusQ = spark.read.parquet(corpusQoutput)
    println("\nAnalyzing Questions")
    lda(corpusQ, spark)
////    println("Questions = " + corpusQ.count())
//    val corpusA = spark.read.parquet(corpusAoutput)
//    println("\nAnalyzing Answers")
//    lda(corpusA, spark)
////    println("Answers = " + corpusA.count())
//    val corpusQA = spark.read.parquet(corpusQAoutput)
//    println("\nAnalyzing Questions + Answers")
//    lda(corpusQA, spark)
////    println("Q n A = " + corpusQA.count())
  }
  
  def lda(corpus : DataFrame, spark: SparkSession) {
    val minTermLenght = 3  // A term should have at least minTermLenght characters
    val qtyOfTopTerms = 20 // how many top terms should be shown in output
    val termMinDocFreq = 3 // minimum number of different documents a term must appear in to be included in the vocabulary
    val qtyLDATopics = 10 // number of LDA latent topics
    val alpha = -1 // choose a low alpha if your documents are made up of a few dominant topics 
    val beta = -1  // choose a low beta if your topics are made up of a few dominant words
    val maxIterations = 30 // number of LDA training iterations
    val termsPerTopic = 20 // how many terms per topic should be shown in output 
    val qtyTopDocPerTopic = 20 // how many top documents per topic should be shown in output
    
    // Tokenization
    val tokenizer = new RegexTokenizer()
      .setPattern("[\\W_]+")
      .setMinTokenLength(minTermLenght) // Filter away tokens with length < minTokenLenght
      .setInputCol("document")
      .setOutputCol("tokens")
    val tokenized_df = tokenizer.transform(corpus)
//    tokenized_df.select("tokens").show(false)
    
    // Removing stopwords
    val stopwords = spark.sparkContext.textFile(stopwordsFile).collect()
    var remover = new StopWordsRemover()
      .setStopWords(stopwords)
      .setInputCol("tokens")
      .setOutputCol("filtered")
    val filtered_df = remover.transform(tokenized_df)
    filtered_df.persist(MEMORY_ONLY)
//    filtered_df.select("filtered").show(false)
    
    // Computing tokens frequencies
    val filtered_df_combined = filtered_df
      .select(org.apache.spark.sql.functions.explode(col("filtered")).alias("filtered_aux"))
      .select(org.apache.spark.sql.functions.collect_list("filtered_aux").alias("filtered")) 
    var vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .fit(filtered_df_combined)
    val countVectors = vectorizer.transform(filtered_df_combined).select("features")
    val frequency = countVectors.rdd.map(_.getAs[SparseVector]("features")).collect()(0)
    println("")
    println("Corpus size = " + corpus.count())
    println("Vocabulary size = " + vectorizer.vocabulary.length)
    println("")
    val tokensFrequency = vectorizer.vocabulary.zip(frequency.toArray)
    println(s"Top $qtyOfTopTerms tokens:")
    tokensFrequency.take(qtyOfTopTerms).foreach(println)
    
    // Computing tokens frequencies for LDA
    vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setMinDF(termMinDocFreq)
      .fit(filtered_df)
    val new_countVectors = vectorizer.transform(filtered_df).select("id", "features")
    val countVectorsMLib = MLUtils.convertVectorColumnsFromML(new_countVectors, "features")
    import spark.implicits._
    val lda_countVector = countVectorsMLib.map { case Row(id: Int, countVector: Vector) => (id.toLong, countVector) }
    
    // LDA
    val lda = new LDA()
      //.setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
      .setOptimizer("em")
      .setK(qtyLDATopics)
      .setMaxIterations(maxIterations)
      .setDocConcentration(alpha) // use default values
      .setTopicConcentration(beta) // use default values
    val ldaModel = lda.run(lda_countVector.rdd)
    
    
    var topicsArray = ldaModel.describeTopics(maxTermsPerTopic = termsPerTopic)
    var vocabList = vectorizer.vocabulary
    var topics = topicsArray.map {
      case (term, termWeight) =>
        term.map(vocabList(_)).zip(termWeight)
      }
    val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
    var topDocs = distLDAModel.topDocumentsPerTopic(qtyTopDocPerTopic)
    
    println("")
    topics.zipWithIndex.foreach {
      case (topic, i) =>
        println(s"TOPIC ${i+1}")
        println("---")
        topic.foreach { 
          case (term, weight) => {
            print(s"$term (")
            print(f"$weight%2.3f) ") 
          }
        }
        println("\n---")
        val temp = (topDocs(i)._1).zip(topDocs(i)._2)
        temp.foreach {
          case (id, weight) => {
            print(f"$weight%2.3f : $id : ")
            println(corpus.where($"id" === id).select("title").first().getAs[String]("title"))
          }
        }
        println(s"==========")
    }
  }
}
