package example

import java.util.Properties

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.deeplearning4j.eval.Evaluation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.apache.spark.sql.functions._
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.apache.log4j.PropertyConfigurator

import scala.collection.JavaConversions._

object SparkWine {
  def main(args: Array[String]): Unit = {
    val props = new Properties()
    props.load(getClass.getResourceAsStream("/log4j-spark.properties"))
    PropertyConfigurator.configure(props)

    // Create a Scala Spark Context.
    val spark = SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate

    import spark.implicits._

    val cols = Array("label", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "pc_of_diluted_wines", "proline")
    // Load CSV files
    val trainingDf = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("./src/main/resources/wine-train.csv")
      .select(cols.map(col(_).cast(DoubleType)) :_*)

    val testDf = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("./src/main/resources/wine-test.csv")
      .select(cols.map(col(_).cast(DoubleType)) :_*)

    val assembler = new VectorAssembler()
      .setInputCols(cols.drop(1))
      .setOutputCol("features")
    val trainFeaturesDf = assembler.transform(trainingDf)
    val testFeaturesDf = assembler.transform(testDf)

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(true)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(trainFeaturesDf)

    // Normalize each feature to have unit standard deviation.
    val scaledtrainingDf = scalerModel.transform(trainFeaturesDf)
      .map( it => LabeledPoint(it.getAs("label"), it.getAs("scaledFeatures")))

    val scaledTestDf = scalerModel.transform(testFeaturesDf)
      .map( it => LabeledPoint(it.getAs("label"), it.getAs("scaledFeatures")))

    val numLinesToSkip = 0
    val delimiter = ','
    val recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReaderTrain.initialize(new FileSplit(new ClassPathResource("wine-train.csv").getFile))
    val recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReaderTest.initialize(new FileSplit(new ClassPathResource("wine-test.csv").getFile))

    // Turn CSVs into DataSets
    val labelIndex = 0
    val numInputs = 13
    val numClasses = 3
    val batchSize = 4

    val epochs = 5
    val seed = 123

    println("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .activation(Activation.RELU)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.1)
      .regularization(true).l2(0.01)
      .updater(new Nesterovs(0.01))
      .list
      .layer(0, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nIn(numInputs).nOut(20).build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(20).nOut(numClasses).build)
      .backprop(true)
      .pretrain(false)
      .build

    val batchSizePerWorker = 4

    // Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
    val tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSizePerWorker) //Async prefetching: 2 examples per worker
      .build

    //Create the Spark network
    val sparkNet = new SparkDl4jMultiLayer(spark.sparkContext, conf, tm)

    implicit val dataSetEncoder = org.apache.spark.sql.Encoders.kryo[DataSet]

    val datasetTrain = scaledtrainingDf.map(fromLabeledPoint(_, numClasses))

    //Execute training:
    (0 to epochs).foreach { i =>
      sparkNet.fit(datasetTrain.javaRDD)
    }

    val datasetTest = scaledTestDf.map(fromLabeledPoint(_, numClasses))

    // evaluate the model on the test set
    val labels = (1 to 3).map { _.toString }.toList
    var eval = new Evaluation(labels)
    val eval2 = sparkNet.doEvaluation(datasetTest.javaRDD, eval, 64)
    println(eval2.stats())
  }

  // Copy of a function from org.deeplearning4j.spark.util.MLLibUtil but using Spark 2.0+ LabeledPoint
  private def fromLabeledPoint(point: LabeledPoint, numPossibleLabels: Int) = {
    val features = point.features
    val label = point.label
    new DataSet(Nd4j.create(features.toArray), FeatureUtil.toOutcomeVector(label.toInt, numPossibleLabels))
  }
}
