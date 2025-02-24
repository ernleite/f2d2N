package com.deeplearning.layer

trait WeightedLayer {
  var minibatch = scala.collection.mutable.HashMap.empty[String, Int]

  var wInitialized: Boolean = false
  var inputs: Array[Float] = Array[Float]()
  var weights = Array[Float]()
  var weighted = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var backPropagateReceived = scala.collection.mutable.HashMap.empty[String, Boolean]
  var messagePropagateReceived = scala.collection.mutable.HashMap.empty[String, Int]
  var fromInternalReceived = scala.collection.mutable.HashMap.empty[String, Int]

  var nablas_w = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var nabla_w = Array[Array[Array[Float]]]()
  var counterTraining: Int = 0
  var counterBackPropagation: Int = 0
  var counterFeedForward: Int = 0
  var debugDelta = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var activation = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var lastEpoch = 0

  def Weights(Epoch: Int, CorrelationId: String, YLabel: Int, trainingCount: Int, Inputs: Array[Float], Shards:Int, InternalSubLayer: Int, Layer: Int, Params : scala.collection.mutable.HashMap[String,String]) : Array[Array[Float]]
  def BackPropagate(CorrelationId: String, delta: Array[Float], LearningRate: Float, Regularisation: Float, nInputs: Int, Layer: Int, InternalSubLayer: Int, FromInternalSubLayer: Int, Params:scala.collection.mutable.HashMap[String,String], ApplyGrads: Boolean) : Boolean
  def FeedForwardTest(CorrelationId: String, Inputs: Array[Float], InternalSubLayer: Int, Layer: Int) : Array[Array[Float]]
  //def SynchronizeWeights(layer:Int, subLayer:Int) : Unit
  def getNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, internalSubLayer:Int) : Unit
  def sendGradients(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int, fromInternalSubLayer:Int): Unit
  def applyGradients(correlationId: String, gradients: Array[Float], layer:Int,internalSubLayerCaller:Int, internalSubLayer:Int): Unit

}
