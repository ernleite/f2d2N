package com.deeplearning.samples

trait TrainingDataSet {
  var trainingSize: Int = 0
  var testSize: Int = 0
  var trainingSetLoaded:Boolean = false
  var testSetLoaded:Boolean = false
  var trainingSet: Array[Array[Float]] = Array[Array[Float]]()
  var testSet: Array[Array[Float]] = Array[Array[Float]]()
  var SourceTrain = ""
  var SourceTest = ""
  var Input2D =""
  var Input2DRows =0
  var Input2DCols =0
  var Channel = 1
  var DatasetName = ""
  var Size = 0

  def loadTrainDataset(loadMode:String): Unit
  def loadTestDataset(loadMode:String): Unit
  def getTrainingInput(index: Int): Array[Float]
  def getTestInput(index: Int): Array[Float]
  def getTrainingLabelData(col:Int): Array[Int]
  def getTestLabelData(col:Int): Array[Int]

}
