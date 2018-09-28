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
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import java.time.LocalDateTime
import java.sql.Timestamp
import org.apache.spark.sql.functions.udf

object LDAExample {

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
    favoriteCount:    Option[Int])

  def parseXml(line: String) = {
    try {
      val xml = scala.xml.XML.loadString(line)
      val id = (xml \@ "Id").toInt
      val postTypeId = (xml \@ "PostTypeId").toInt
      val creationDate = Timestamp.valueOf(LocalDateTime.parse(xml \@ "CreationDate"))
//      val creationDate = LocalDateTime.parse(xml \@ "CreationDate")
      val score = (xml \@ "Score").toInt
      val body = (xml \@ "Body")
      
//        .toLowerCase()
//      val body = scala.xml.XML.loadString(xml \@ "Body")
//        .text // remove html tags
//        .filter(_ >= ' ') // throw away all control characters.
//        .toLowerCase()
      
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
          favoriteCount)
          )
    } catch {
      case e: Exception => None
    }
  }
  
  def isSparkRelated(tags : String) = {
    tags.contains("apache-spark") || 
    tags.contains("pyspark") ||
    tags.contains("spark-dataframe") ||
    tags.contains("spark-streaming") ||
    tags.contains("sparkr") ||
    tags.contains("spark-cassandra-connector") ||
    tags.contains("sparklyr") ||
    tags.contains("spark-graphx") ||
    tags.contains("spark-submit") ||
    tags.contains("spark-structured-streaming") ||
    tags.contains("spark-csv") ||
    tags.contains("spark-avro")
  }

  def main(args: Array[String]) {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("LDAExample")
//      .master("local[*]")
      .getOrCreate()

//    val lines = spark.sparkContext.textFile("./resources/Posts-Spark.xml").flatMap(parseXml)
//    val lines = spark.sparkContext.textFile("./resources/Posts-Spark-100.xml").flatMap(parseXml)
    val lines = spark.sparkContext.textFile("hdfs://master:54310/user/hduser/stackoverflow/Posts.xml").flatMap(parseXml)
    
    import spark.implicits._

    // Filtrar Posts para somente aqueles em que eh possivel ter pergunta sobre Apache Spark
    val posts = lines
      .toDS()
      .where("year(creationDate) > 2012")
  
    // Filtrar Posts para somente os que tiverem tags relacionadas a Apache Spark. 
    // Somente perguntas possuem Tags
    spark.udf.register("sparkRelated", (tags : String) =>  isSparkRelated(tags))
    val sparkQuestions = posts
      .where("postTypeId = 1")
      .withColumn("sparkRelated", expr("sparkRelated(tags)"))
      .where("sparkRelated")
    sparkQuestions.printSchema()
    println("Count Questions = " + sparkQuestions.count)
    
//    val c = posts.count
//    println("Count Posts = " + c)
        
    //
    ////    println("Count = " + posts.count())
    //
    //    val corpus = posts.withColumn("document", concat($"title", lit(" "), $"body")).select("id","document")
    //
    ////    corpus.show(false)
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
    //    }

    spark.stop()
  }

}