package com.deeplearning.layer
trait ActivationLayer {
  var minibatch = scala.collection.mutable.HashMap.empty[String, Int]
  var messagePropagateReceived = scala.collection.mutable.HashMap.empty[String, Int]

  var bInitialized: Boolean = false
  var shardReceived = scala.collection.mutable.HashMap.empty[String, Int]
  var inProgress = scala.collection.mutable.HashMap.empty[String, Boolean]
  var backPropagateReceived = scala.collection.mutable.HashMap.empty[String, Boolean]
  var weighted = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var bias = Array[Float]()
  var activation = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var Z = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var nablas_b = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var nabla_b = Array[Float]()
  var counterTraining: Int = 0
  var counterBackPropagation: Int = 0
  var counterFeedForward: Int = 0
  var lastEpoch = 0

  def ComputeZ(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, shardedWeighted: Array[Float], internalSubLayer: Int, fromInternalSubLayer:Int, layer: Int, shards: Int, params:scala.collection.mutable.HashMap[String,String]) : Array[Float]
  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, shards:Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer:Int, params : scala.collection.mutable.HashMap[String,String]) : Boolean
  def FeedForwardTest(correlationId: String, weighted: Array[Float], internalSubLayer: Int,fromInternalSubLayer:Int, layer: Int, shards: Int) : Array[Float]
  def getNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, internalSubLayer:Int) : Unit
  def sendGradients(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int, fromInternalSubLayer:Int): Unit
  def applyGradients(correlationId: String, gradients: Array[Float], layer:Int,internalSubLayerCaller:Int, internalSubLayer:Int): Unit
}
