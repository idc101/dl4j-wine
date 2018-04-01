package example

import scala.collection.JavaConversions._
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.datavec.api.util.ClassPathResource
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions



object WineTaster {
  def main(args: Array[String]): Unit = {
    //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
    val numLinesToSkip = 1
    val delimiter = ';'
    val recordReaderTrain = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReaderTrain.initialize(new FileSplit(new ClassPathResource("winequality-red-train.csv").getFile))
    val recordReaderTest = new CSVRecordReader(numLinesToSkip, delimiter)
    recordReaderTest.initialize(new FileSplit(new ClassPathResource("winequality-red-test.csv").getFile))

    //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
    val labelIndex = 11
    val numInputs = labelIndex
    val numClasses = 10
    val batchSize = 4

    val iteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, labelIndex, numClasses)
    val iteratorTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, labelIndex, numClasses)


    //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
    val normalizer = new NormalizerStandardize
    normalizer.fit(iteratorTrain) //Collect the statistics (mean/stdev) from the training data. This does not modify the input data

    iteratorTrain.setPreProcessor(normalizer)
    iteratorTest.setPreProcessor(normalizer)

//    val next = iteratorTrain.next()
//    import org.nd4j.linalg.factory.Nd4j
//    Nd4j.writeTxtString(next.getFeatureMatrix, System.out, 3)
//    iteratorTrain.reset()

    val epochs = 5
    val iterations = 1
    val seed = 123


    println("Build model....")
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation(Activation.RELU)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.01)
      .regularization(true).l2(0.01)
      .updater(new Nesterovs(0.1))
      .list
      .layer(0, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nIn(numInputs).nOut(40).build)
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nIn(40).nOut(numClasses).build)
      .backprop(true)
      .pretrain(false)
      .build

    // run the model
    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(5))

    //Initialize the user interface backend
    val uiServer = UIServer.getInstance()

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage()         //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    //Then add the StatsListener to collect this information from the network, as it trains
    model.setListeners(new StatsListener(statsStorage, 10))

    // train the model
    val epochsIterator = new MultipleEpochsIterator(epochs, iteratorTrain)
    model.fit(epochsIterator)

    // evaluate the model on the test set
    val labels = (1 to 10).map { _.toString }.toList
    val eval = model.evaluate(iteratorTest, labels)
    println(eval.stats())
  }
}
