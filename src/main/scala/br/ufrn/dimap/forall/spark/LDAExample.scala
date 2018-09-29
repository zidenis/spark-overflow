package br.ufrn.dimap.forall.spark

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.clustering.OnlineLDAOptimizer
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import java.time.LocalDateTime
import java.sql.Timestamp

object LDAExample {
  
  val resourceInput = "./resources/Posts-Spark-100.xml"
  val corpusQoutput = "./resources/CorpusQ.parquet"
  val corpusAoutput = "./resources/CorpusA.parquet"
  val corpusQAoutput = "./resources/CorpusQA.parquet"
//    val resourceInput = "hdfs://master:54310/user/hduser/stackoverflow/Posts.xml"
//    val corpusQoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQ.parquet"
//    val corpusAoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusA.parquet"
//    val corpusQAoutput = "hdfs://master:54310/user/hduser/stackoverflow/CorpusQA.parquet"
  
  val spark = SparkSession
      .builder
      .appName("LDAExample")
      .master("local[*]")
      .getOrCreate()
  
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
  
  def reading() {
    val corpusQ = spark.read.format("parquet").parquet(corpusQoutput)
    println("Questions = " + corpusQ.count())
    val corpusA = spark.read.format("parquet").parquet(corpusAoutput)
    println("Answers = " + corpusA.count())
    val corpusQA = spark.read.format("parquet").parquet(corpusQAoutput)
    println("Q n A = " + corpusQA.count())
  }

  def main(args: Array[String]) {
    
    Logger.getLogger("org").setLevel(Level.ERROR) // Set the log level to only print errors
    processing()
    reading()
    spark.stop()
  }
  
  def processing() {
  
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
    corpusQ.createOrReplaceTempView("corpusQ")
//    println("Questions = " + corpusQ.count())
//    corpusQ.show(10, false)
    corpusQ.write.mode(SaveMode.Overwrite).format("parquet").save(corpusQoutput)

    // Obter Posts com respostas a perguntas sobre Spark
    val stackAnswers = posts
      .where("postTypeId = 2") // somente respostas
    stackAnswers.createOrReplaceTempView("stackAnswers")
//    val sparkAnswers = stackAnswers
//      .join(sparkQuestions, stackAnswers("parentId") === sparkQuestions("id"), "leftsemi")
    val corpusA = sql("""
      SELECT a.parentId
           , cleanDocument(concat_ws(' ', collect_list(a.body))) as document 
        FROM stackAnswers a 
   LEFT SEMI JOIN corpusQ q 
          ON a.parentId = q.id 
    GROUP BY a.parentId
    """)
    corpusA.createOrReplaceTempView("corpusA")
//    println("Answers = " + corpusA.count())
//    corpusA.show(10, false)
    corpusA.write.mode(SaveMode.Overwrite).format("parquet").save(corpusAoutput)
    
    // Obter Posts com perguntas sobre Spark e suas respectivas perguntas
    val sparkQA = sql("""
      SELECT id, q.document qd, a.document ad 
        FROM corpusQ q 
   LEFT JOIN corpusA a 
          ON q.id = a.parentId
    """)
    val corpusQA = sparkQA
      .withColumn("ad_not_null", coalesce($"ad",lit("")))
      .withColumn("document", concat($"qd", lit(" "), $"ad_not_null"))
      .select("id","document")
//    println("QA = " + corpusQA.count())
//    corpusQA.show(10, false)
    corpusQA.write.mode(SaveMode.Overwrite).format("parquet").save(corpusQAoutput)
  }
  
  def lda() {  
    //
    //    // Set params for RegexTokenizer
    //    val tokenizer = new RegexTokenizer()
    //      .setPattern("[\\W_]+")
    //      .setMinTokenLength(4) // Filter away tokens with length < 3, no original era 4, mas coloquei 3 por causa do termo RDD
    //      .setInputCol("document") // acho que nao deveriamos desprezar o title
    //      .setOutputCol("tokens")
    //
    //    // Tokenize document
    //    val tokenized_df = tokenizer.transform(corpus)
    //
    ////    tokenized_df.select("tokens").show(false)
    ////    tokenized_df.show()
    //
    //    // List of stopwords
    //    val stopwords = spark.sparkContext.textFile("./resources/stopwords.txt").collect()
    //
    //    // Set params for StopWordsRemover
    //    var remover = new StopWordsRemover()
    //      .setStopWords(stopwords) // This parameter is optional
    //      .setInputCol("tokens")
    //      .setOutputCol("filtered")
    //
    //    // Create new DF with Stopwords removed
    //    val filtered_df = remover.transform(tokenized_df)
    //
    ////    filtered_df.select("filtered").show(false)
    //
    //    // Set params for CountVectorizer
    //    var vectorizer = new CountVectorizer()
    //      .setInputCol("filtered")
    //      .setOutputCol("features")
    //      .setVocabSize(300)
    //      .setMinDF(1)
    //      .fit(filtered_df)
    //
    //    // Create vector of token counts
    //    val countVectors = vectorizer.transform(filtered_df).select("id", "features")
    //
    ////    countVectors.show(false)
    //
    //    val countVectorsMLib = MLUtils.convertVectorColumnsFromML(countVectors, "features")
    //
    //    val lda_countVector = countVectorsMLib.map { case Row(id: Long, countVector: Vector) => (id, countVector) }
    //
    //    val numTopics = 10
    //
    //    // Set LDA params
    //    val lda = new LDA()
    //      .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
    //      .setK(numTopics)
    //      .setMaxIterations(3)
    //      .setDocConcentration(-1) // use default values
    //      .setTopicConcentration(-1) // use default values
    //
    //    val ldaModel = lda.run(lda_countVector.rdd)
    //
    //    // Review Results of LDA model with Online Variational Bayes
    //    var topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    //    var vocabList = vectorizer.vocabulary
    //    var topics = topicIndices.map {
    //      case (terms, termWeights) =>
    //        terms.map(vocabList(_)).zip(termWeights)
    //    }
    //    println("First Topics")
    //    println(s"$numTopics topics:")
    //    topics.zipWithIndex.foreach {
    //      case (topic, i) =>
    //        println(s"TOPIC $i")
    //        topic.foreach { case (term, weight) => println(s"$term\t$weight") }
    //        println(s"==========")
    //    }

    //    val add_stopwords = Array("code", "pre", "true", "false", "string", "int", "apache", "org", "using", "com", "github", "import", "new", "info", "2018", "artifactid", "groupid", "apply", "anonfun", "val", "var", "version", "jar", "href", "https")
    //
    //    // Combine newly identified stopwords to our exising list of stopwords
    //    val new_stopwords = stopwords.union(add_stopwords)
    //
    //    // Set Params for StopWordsRemover with new_stopwords
    //    remover = new StopWordsRemover()
    //      .setStopWords(new_stopwords)
    //      .setInputCol("tokens")
    //      .setOutputCol("filtered")
    //
    //    // Create new df with new list of stopwords removed
    //    val new_filtered_df = remover.transform(tokenized_df)
    //
    //    // Set Params for CountVectorizer
    //    vectorizer = new CountVectorizer()
    //      .setInputCol("filtered")
    //      .setOutputCol("features")
    //      .setVocabSize(10000)
    //      .setMinDF(5)
    //      .fit(new_filtered_df)
    //
    //    // Create new df of countVectors
    //    val new_countVectors = vectorizer.transform(new_filtered_df).select("id", "features")
    //
    //    val new_countVectorsMLib = MLUtils.convertVectorColumnsFromML(new_countVectors, "features")
    //
    //    // Convert DF to RDD
    //    val new_lda_countVector = new_countVectorsMLib.map { case Row(id: Long, countVector: Vector) => (id, countVector) }
    //
    //    // Set LDA parameters
    //    val new_lda = new LDA()
    //      .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
    //      .setK(numTopics)
    //      .setMaxIterations(10)
    //      .setDocConcentration(-1) // use default values
    //      .setTopicConcentration(-1) // use default values
    //
    //    // Create LDA model with stopwords refiltered
    //    val new_ldaModel = new_lda.run(new_lda_countVector.rdd)
    //
    //    topicIndices = new_ldaModel.describeTopics(maxTermsPerTopic = 10)
    //    vocabList = vectorizer.vocabulary
    //    topics = topicIndices.map {
    //      case (terms, termWeights) =>
    //        terms.map(vocabList(_)).zip(termWeights)
    //    }
    //    println("Topics after remove some words:")
    //    println(s"$numTopics topics:")
    //    topics.zipWithIndex.foreach {
    //      case (topic, i) =>
    //        println(s"TOPIC $i")
    //        topic.foreach { case (term, weight) => println(s"$term\t$weight") }
    //        println(s"==========")
  }
  
}