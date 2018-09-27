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
import org.apache.spark.sql.functions.{concat, lit}

object LDAExample {

  case class Post(id: Long, postTypeId: Int, title: String, body: String, tags: String, parentId: Long)

  def parseXml(line: String) = {
    try {
      val xml = scala.xml.XML.loadString(line)
      val Id = (xml \@ "Id").toLong
      val PostTypeId = (xml \@ "PostTypeId").toInt
      val Body = scala.xml.XML.loadString(xml \@ "Body")
        .text // remove html tags
        .filter(_ >= ' ') // throw away all control characters.
        .toLowerCase()
      val Title = xml \@ "Title"
      val Tags = xml \@ "Tags"
      var ParentId = Id
      if (PostTypeId == 2) {
        ParentId = (xml \@ "ParentId").toLong 
      } 
      
      Some(Post(Id, PostTypeId, Title, Body, Tags, ParentId))
    } catch {
      case e: Exception => None
    }
  }

  def main(args: Array[String]) {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    // Use new SparkSession interface in Spark 2.0
    val spark = SparkSession
      .builder
      .appName("LDAExample")
      .master("local[*]")
      //.config("spark.sql.warehouse.dir", "file:///C:/temp") // Necessary to work around a Windows bug in Spark 2.0.0; omit if you're not on Windows.
      .getOrCreate()

    val lines = spark.sparkContext.textFile("./resources/Posts-Spark-100.xml").flatMap(parseXml)
  
    //import org.apache.spark.sql.SparkSession.implicits._
    import spark.implicits._
    
    val posts = lines.toDS()
    
//    posts.printSchema()
//    posts.show
//    posts.show(60,false)
      
//    println("Count = " + posts.count())
    
    val corpus = posts.withColumn("document", concat($"title", lit(" "), $"body")).select("id","document")
    
//    corpus.show(false)

    // Set params for RegexTokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("[\\W_]+")
      .setMinTokenLength(4) // Filter away tokens with length < 3, no original era 4, mas coloquei 3 por causa do termo RDD
      .setInputCol("document") // acho que nao deveriamos desprezar o title
      .setOutputCol("tokens") 

    // Tokenize document
    val tokenized_df = tokenizer.transform(corpus)

//    tokenized_df.select("tokens").show(false)
//    tokenized_df.show()

    // List of stopwords
    val stopwords = spark.sparkContext.textFile("./resources/stopwords.txt").collect()

    // Set params for StopWordsRemover
    var remover = new StopWordsRemover()
      .setStopWords(stopwords) // This parameter is optional
      .setInputCol("tokens")
      .setOutputCol("filtered")

    // Create new DF with Stopwords removed
    val filtered_df = remover.transform(tokenized_df)

//    filtered_df.select("filtered").show(false)
    
    // Set params for CountVectorizer
    var vectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("features")
      .setVocabSize(300)
      .setMinDF(1)
      .fit(filtered_df)

    // Create vector of token counts
    val countVectors = vectorizer.transform(filtered_df).select("id", "features")
    
//    countVectors.show(false)

    val countVectorsMLib = MLUtils.convertVectorColumnsFromML(countVectors, "features")

    val lda_countVector = countVectorsMLib.map { case Row(id: Long, countVector: Vector) => (id, countVector) }

    val numTopics = 10

    // Set LDA params
    val lda = new LDA()
      .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
      .setK(numTopics)
      .setMaxIterations(3)
      .setDocConcentration(-1) // use default values
      .setTopicConcentration(-1) // use default values

    val ldaModel = lda.run(lda_countVector.rdd)

    // Review Results of LDA model with Online Variational Bayes
    var topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    var vocabList = vectorizer.vocabulary
    var topics = topicIndices.map {
      case (terms, termWeights) =>
        terms.map(vocabList(_)).zip(termWeights)
    }
    println("First Topics")
    println(s"$numTopics topics:")
    topics.zipWithIndex.foreach {
      case (topic, i) =>
        println(s"TOPIC $i")
        topic.foreach { case (term, weight) => println(s"$term\t$weight") }
        println(s"==========")
    }

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