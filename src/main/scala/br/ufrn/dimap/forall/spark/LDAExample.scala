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
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel._
import java.time.LocalDateTime
import java.sql.Timestamp

object LDAExample {
  
//  val resourceInput = "./resources/Posts-Spark-100.xml"
//  val corpusQoutput = "./resources/CorpusQ.parquet"
//  val corpusAoutput = "./resources/CorpusA.parquet"
//  val corpusQAoutput = "./resources/CorpusQA.parquet"
//  val stopwordsFile = "./resources/stopwords.txt"
  
  val resourceInput = "hdfs://master:54310/user/hduser/stackoverflow/Posts.xml"
  val corpusQoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQ.parquet"
  val corpusAoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusA.parquet"
  val corpusQAoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQA.parquet"
  val stopwordsFile = "hdfs://master:54310/user/hduser/stackoverflow/stopwords.txt"
  
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR) // Set the log level to only print errors
    val spark = SparkSession
      .builder
      .appName("LDAExample")
//      .master("local[*]")
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
      .replaceAll("<pre>.+</pre>", " CODE ") // remove code parts
      .replaceAll("<([^>])+>", " TAG ") // remove tags
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
      .select("id","document")
    corpusQ.persist(MEMORY_ONLY)
    corpusQ.createOrReplaceTempView("corpusQ")
//    println("Questions = " + corpusQ.count())
//    corpusQ.show(10, false)
    corpusQ.write.mode(SaveMode.Overwrite).parquet(corpusQoutput)

    // Obter Posts com respostas a perguntas sobre Spark
    val stackAnswers = posts
      .where("postTypeId = 2") // somente respostas
    stackAnswers.createOrReplaceTempView("stackAnswers")
//    val sparkAnswers = stackAnswers
//      .join(sparkQuestions, stackAnswers("parentId") === sparkQuestions("id"), "leftsemi")
    val corpusA = sql("""
      SELECT a.parentId as id
           , cleanDocument(concat_ws(' ', collect_list(a.body))) as document 
        FROM stackAnswers a 
   LEFT SEMI JOIN corpusQ q 
          ON a.parentId = q.id 
    GROUP BY a.parentId
    """)
    corpusA.persist(MEMORY_ONLY)
    corpusA.createOrReplaceTempView("corpusA")
//    println("Answers = " + corpusA.count())
//    corpusA.show(10, false)
    corpusA.write.mode(SaveMode.Overwrite)parquet(corpusAoutput)
    
    // Obter Posts com perguntas sobre Spark e suas respectivas perguntas
    val sparkQA = sql("""
      SELECT id, q.document qd, a.document ad 
        FROM corpusQ q 
   LEFT JOIN corpusA a 
          ON q.id = a.id
    """)
    val corpusQA = sparkQA
      .withColumn("ad_not_null", coalesce($"ad",lit("")))
      .withColumn("document", concat($"qd", lit(" "), $"ad_not_null"))
      .select("id","document")
//    println("QA = " + corpusQA.count())
//    corpusQA.show(10, false)
    corpusQA.write.mode(SaveMode.Overwrite).parquet(corpusQAoutput)
  }
  
  def reading(spark: SparkSession) {
    val corpusQ = spark.read.parquet(corpusQoutput)
    println("\nAnalyzing Questions")
    lda(corpusQ, spark)
//    println("Questions = " + corpusQ.count())
    val corpusA = spark.read.parquet(corpusAoutput)
    println("\nAnalyzing Answers")
    lda(corpusA, spark)
//    println("Answers = " + corpusA.count())
    val corpusQA = spark.read.parquet(corpusQAoutput)
    println("\nAnalyzing Questions + Answers")
    lda(corpusQA, spark)
//    println("Q n A = " + corpusQA.count())
  }
  
  def lda(corpus : DataFrame, spark: SparkSession) {
     
    // Tokenization
    val tokenizer = new RegexTokenizer()
      .setPattern("[\\W_]+")
      .setMinTokenLength(3) // Filter away tokens with length < 3, no original era 4, mas coloquei 3 por causa do termo RDD
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
    println("Vocabulary total size = " + vectorizer.vocabulary.length)
    val tokensFrequency = vectorizer.vocabulary.zip(frequency.toArray)
    println("Top 100 Tokens:")
    tokensFrequency.take(100).foreach(println)
    
    // Computing tokens frequencies for LDA
    vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setMinDF(3) // Tokens utilizados pelo menos em 2 documentos
      .fit(filtered_df)
    val new_countVectors = vectorizer.transform(filtered_df).select("id", "features")
    val countVectorsMLib = MLUtils.convertVectorColumnsFromML(new_countVectors, "features")
    import spark.implicits._
    val lda_countVector = countVectorsMLib.map { case Row(id: Int, countVector: Vector) => (id.toLong, countVector) }
    
    // LDA
    val numTopics = 10
    val termsPerTopic = 20
    val lda = new LDA()
      .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
      .setK(numTopics)
      .setMaxIterations(3)
      .setDocConcentration(-1) // use default values
      .setTopicConcentration(-1) // use default values
    val ldaModel = lda.run(lda_countVector.rdd)
    var topicIndices = ldaModel.describeTopics(maxTermsPerTopic = termsPerTopic)
    var vocabList = vectorizer.vocabulary
    var topics = topicIndices.map {
      case (terms, termWeights) =>
        terms.map(vocabList(_)).zip(termWeights)
      }
    
    println("")
    topics.zipWithIndex.foreach {
      case (topic, i) =>
        println(s"TOPIC ${i+1}")
        topic.foreach { 
          case (term, weight) => {
            print(s"$term\t")
            if (term.length() < 9) print("\t")
            println(s"$weight") 
          }
        }
        println(s"==========")
    }
    
  }
}
