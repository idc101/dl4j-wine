## Neural network for Wine classification using DL4J

### Data

Data is from UCI:
https://archive.ics.uci.edu/ml/datasets/wine

Attempts to classify the region a wine came from based on
it's chemical properties

### Wine

`Wine.scala` uses DL4J to read the data and train the model.


### SparkWine

`SparkWine.scala` uses Spark to read the data as DataFrames and
Spark ML pipelines to normalize the data then uses DL4J to train
the model.
