package com.deeplearning

import breeze.linalg.{DenseVector}
import breeze.numerics.sqrt
import breeze.stats.distributions.{Gaussian}
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import scala.util.Random

object Network {

  val clusterNodesDim = 1 // Cluster dimension
  val trainingSample = "Cifar10" // Sample Mnist or Cifar10
  val channels = if (Network.trainingSample == "Cifar10") 3 else 1 //1 Mnist or 3 Cifar
  val InputLayerType = "Dense"
  val InputActivationType = "Relu"
  val InputLayer = if (Network.trainingSample == "Cifar10") 3072 else 784 // 784 Mnist or 3072 Cifar
  val InputLayerDim = 6 // Vertical split
  val HiddenLayers = Array[Int](500,250) // Neurons size : horizontal parallelism
  val HiddenLayersDim = Array[Int](5,2) // Vertical parallelism : Disabled if 1
  val HiddenLayerType = Array[String]( "Dense","Dense","Dense","Dense") // Dense or Conv2D
  val HiddenActivationType = Array[String]( "Relu", "Relu", "Sigmoid", "Sigmoid") // Sigmoid, Relu, TanH, LeakyRelu
  val Filters = Array[String]("filters:5;kernel:3,3;stride:1;padding:same","filters:10;kernel:3,3;stride:1;padding:same")
  val ForwardBatchNormalization=false
  val BackBatchNormalization=false
  val OutputLayer = 10
  val OutputLayerDim = 1
  val OutputActivationType = "SoftMax"

  val CostFunction = "CategoricalCrossEntropy"
  var LearningRate:Float =  0.195f
  val weightedPenalty = 0.05f
  var InitialLearningRate:Float = LearningRate
  var Regularisation:Float = 5f
  val limitedDataset = false
  val LayerNorm = true
  val SGB = false
  val Epochs = if (Network.limitedDataset) 1 else 20
  var MiniBatch:Int = if (Network.limitedDataset) 1 else 50
  val MiniBatchRange:Int = if (Network.limitedDataset) 2 else if (Network.trainingSample == "Cifar10") 45000 else 60000 // Mnist 60000 or Cifar 45000
  val minibatchBuffer = if (Network.limitedDataset) 1 else if (Network.trainingSample == "Cifar10") 45000 else 60000 // <= MiniBatchRange
  var rangeInitAuto:Boolean = true
  var rangeInitStart:Float = -1f
  var rangeInitEnd:Float = 1f
  var scaleInitStart: Float = -1f
  var scaleInitEnd: Float = 1f
  var LeakyReluAlpha:Float = 0.01f
  var NaN:Boolean = false
  var CheckNaN:Boolean = false
  var dropout:Float = 0.5f//-1 Dropout desactivated
  val drop = 0.325f
  val epochs_drop = 15
  val debug:Boolean = false
  val debugActivity:Boolean = false
  val debugLevel:Int = 4
  val InputLoadMode = "local"
  val GpuMode = false
  val StatEvents = false

  val debugDelay = false
  val LearningRateDecay = false //enable reducing learning rate when reaching a threshold
  val LearningRateDecayValue = false //enable reducing learning rate when reaching a threshold
  val autoWeigthNormalisation = true
  val autoActivationNormalisation = true

  var Layers = scala.collection.mutable.HashMap.empty[String, String]
  var LayersIntermediateRef = scala.collection.mutable.HashMap.empty[String, akka.actor.typed.ActorRef[com.deeplearning.ComputeWeighted.WeightedCommand]]
  var LayersHiddenRef = scala.collection.mutable.HashMap.empty[String, akka.actor.typed.ActorRef[com.deeplearning.ComputeActivation.ActivationCommand]]
  var LayersInputRef = scala.collection.mutable.HashMap.empty[String, akka.actor.typed.ActorRef[com.deeplearning.ComputeInputs.InputCommand]]
  var LayersOutputRef = scala.collection.mutable.HashMap.empty[String, akka.actor.typed.ActorRef[com.deeplearning.ComputeOutput.OutputCommand]]
  var EpochsRef = scala.collection.mutable.HashMap.empty[String, akka.actor.typed.ActorRef[com.deeplearning.ComputeEpochs.TrainCommand]]

  def getActivationLayersType(layerIndex: Int): String = {
    if (layerIndex  == 1) {
      Network.InputActivationType
    }
    else {
      Network.HiddenActivationType( (layerIndex -1)/2 )
    }
  }

  def getLayersDim(layerIndex: Int): Int = {
    if (layerIndex == 0)
      Network.InputLayerDim
    else
      Network.HiddenLayersDim(layerIndex - 1)
  }

  def getHiddenLayers(layerIndex: Int, layer:String): Int = {
    var indx = 0
    if (layer == "weighted")
      indx = layerIndex/2 -1
    else
      indx = (layerIndex+1 )/2 -1

    Network.HiddenLayers(indx)
  }

  def getHiddenLayersDim(layerIndex:Int, layer:String): Int = {
    var indx = 0
    if (layer == "weighted")
      indx = layerIndex/2 -1
    else
      indx = (layerIndex+1 )/2 -1

    Network.HiddenLayersDim(indx)
  }
  def getHiddenLayersType(layerIndex: Int, layer:String): String = {
    var indx = 0
    if (layer == "weighted")
      indx = layerIndex/2 -1
    else
      indx = (layerIndex+1 )/2 -1

    Network.HiddenLayerType(indx)
  }


  def stepDecay(epoch: Int, initial_lr: Float, drop: Float, epochs_drop: Int): Float = {
    val lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    lr.toFloat
  }
  def shuffleArray(arr: Array[Float]): Array[Float] = {
    val random = new Random()
    for (i <- arr.indices.reverse) {
      val j = random.nextInt(i + 1)
      val temp = arr(i)
      arr(i) = arr(j)
      arr(j) = temp
    }
    arr
  }

  def generateArray(start: Int, end: Int): Array[Int] = {
    val list = (start to end).toList
    if (Network.SGB)
      Random.shuffle(list).toArray
    else
      list.toArray
  }

  def generateUniformRandomFloat(size: Int): Array[Float] = {
    val rand = new Random()
    val stdDev = sqrt( size).toFloat // He initialization
    Array.fill[Float](size)(rand.nextGaussian().toFloat * stdDev)
  }

  def generateRandomBiasFloat(step: Int): Array[Float] = {
    Array.fill(step)(0.01f)
  }

  def generateFixedFloat(step: Int): Array[Float] = {
    val random = new Random()
    Array.fill(step)(0.01f)
  }

  def generateRandomFloat(step: Int): Array[Float] = {
    val random = new Random()
    random.setSeed(100)
    Array.fill[Float](step)(random.between(Network.rangeInitStart, Network.rangeInitEnd))
  }

  def heInitialization(rows:Int, cols:Int, slice:Int): Array[Float] = {
    //val weightMatrix = (DenseMatrix.rand[Float](rows, cols) * math.sqrt(variance).toFloat )
    val fanIn = rows * cols * slice
    val stddev = math.sqrt(2.0 / fanIn)

    val gaussian = new Gaussian(0.0, stddev)
    val data = gaussian.sample(rows * cols).map(_.toFloat)
    //val normalized = CostManager.scalling(data.toArray,slice, Network.rangeInitStart, Network.rangeInitEnd)
    //normalized
    data.toArray
  }

  def generateRandomFloat(step:Int,layer:Int) : Array[Float] = {
    val random = new Random()
    Array.fill[Float](step)(random.between(Network.rangeInitStart,Network.rangeInitEnd))
  }

  //xavier init

  def dropout(input: Array[Float]): Array[Float] = {
    val rng = new Random()
    input.map(elem => if (rng.nextFloat() < Network.dropout) elem / Network.dropout else 0.0f)
  }
  def replaceNanWithZero(vec: DenseVector[Float]): Array[Float] = {
    val replaced = vec.mapValues(v => if (v.isNaN) 0.0f else v)
    replaced.toArray
  }



}