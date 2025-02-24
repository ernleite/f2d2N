package com.deeplearning.layer

trait InputLayer {
  var wInitialized: Boolean = false
  var wTest: Boolean = false
  var weights = Array[Float]()
  var weighted = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var minibatch = scala.collection.mutable.HashMap.empty[String, Int]
  var backPropagateReceived = scala.collection.mutable.HashMap.empty[String, Boolean]
  var X = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var XTest = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var nablas_w = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var nabla_w = Array[Array[Array[Float]]]()
  var counterTraining: Int = 0
  var counterBackPropagation: Int = 0
  var counterFeedForward: Int = 0
  var epochCounter = 0
  var lastEpoch = 0

  def computeInputWeights(epoch: Int, correlationId: String, yLabel:Int, startIndex: Int, endIndex: Int, data: Array[Float], index: Int, layer: Int, internalSubLayer: Int, Params: scala.collection.mutable.HashMap[String,String]): Array[Array[Float]]
  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Float, internalSubLayer: Int, fromInternalSubLayer: Int, Params: scala.collection.mutable.HashMap[String,String]): Boolean
  def FeedForwardTest(correlationId: String, startIndex: Int, endIndex: Int, index: Int, data: Array[Float], internalSubLayer: Int, layer: Int) :  Array[Array[Float]]
}
