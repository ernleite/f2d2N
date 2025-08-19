package com.deeplearning.layer

import com.deeplearning.Network.{generateFixedFloat, generateRandomFloat}
import com.deeplearning.{ComputeActivation, MatrixHelper, LayerManager, Network}
import com.deeplearning.samples.{CifarData, MnistData, TrainingDataSet}

import java.time.{Duration, Instant}

class DenseInputLayer extends InputLayer {
  var parameterSended = false
  var deltas2 = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var nablas_w_tmp2 = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  /*
  private var dataSet: TrainingDataSet = if (Network.trainingSample == "Mnist") {
    println("Loading Dataset MINST")
    dataSet = new MnistData()
    dataSet
  } else {
    dataSet = new CifarData()
    dataSet
  }
   */

  def computeInputWeights(epoch:Int, correlationId: String, yLabel:Int, startIndex:Int, endIndex:Int, data: Array[Float], index:Int, layer:Int, internalSubLayer:Int,params: scala.collection.mutable.HashMap[String,String]): Array[Array[Float]] = {
    if (lastEpoch != epoch) {
      counterTraining = 0
      counterBackPropagation = 0
      counterFeedForward = 0
      lastEpoch = epoch
    }
    epochCounter = epoch
    counterTraining += 1
    val startedAt = Instant.now
    val nextLayer = layer+1
    if (!wInitialized) {
     // dataSet.loadTrainDataset(Network.InputLoadMode)
      weights = generateFixedFloat(Network.getHiddenLayers(nextLayer, "hidden")*(endIndex-startIndex))
      wInitialized = true
    }
    var startAt = Instant.now
    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      //val input = dataSet.getTrainingInput(index)
      //val x = input.slice(startIndex + 2, endIndex + 2) //normalisation
      this.X += (correlationId -> data)
    }
    var endedAt = Instant.now
    var duration = Duration.between(startedAt, endedAt).toMillis
    startAt = Instant.now

    val w1 = MatrixHelper.matrixMult(weights, this.X(correlationId), Network.getHiddenLayers(nextLayer, "hidden"))
    endedAt = Instant.now
    duration = Duration.between(startedAt, endedAt).toMillis

    if (!parameterSended) {
    //  parameters("min") = weights.min.toString
    //  parameters("max") = weights.max.toString
      parameterSended = true
    }

    weighted(correlationId) = w1
    val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_0")
    actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, Network.MiniBatchRange, w1, 0, internalSubLayer, nextLayer, Network.InputLayerDim, params)
    null
  }

  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Float, internalSubLayer: Int, fromInternalSubLayer: Int, params: scala.collection.mutable.HashMap[String,String]): Boolean = {
    counterBackPropagation += 1
    minibatch(correlationId) += 1
    //params("eventBP") =  (params("eventBP").toInt + 1).toString

    if (!backPropagateReceived.contains(correlationId)) {
      val fromArraySize = Network.getHiddenLayersDim(1, "hidden")
      backPropagateReceived += (correlationId -> true)
      //nablas_w(correlationId) = Array.ofDim(fromArraySize)
    }
    var startedAt = Instant.now
    deltas2 += (correlationId -> delta)
    val callerSize = 1
    // check if we reach the last mini-bacth
    //context.log.info("Receiving from bakpropagation")
    //if (minibatch.values.sum >298) println(minibatch.values.sum)
    if ((Network.MiniBatch ) == minibatch.values.sum) {
      parameterSended = false
      params("min") = "0"
      params("max") = "0"

      val fromUCs = 1
      //val weightsDelta = this.weights
      //val split = weightsDelta.grouped(this.weights.size/fromUCs).toArray

      val act = this.X.values.flatten.toArray
      val del = deltas2.values.flatten.toArray
      weights= MatrixHelper.applyGradients(del,act, Network.MiniBatch , learningRate / Network.MiniBatch,1 - learningRate * (regularisation / nInputs),this.weights)

      //weights = split.flatten

      deltas2.clear()
      nablas_w_tmp2.clear()
      backPropagateReceived.clear()
      minibatch.clear()
      weighted.clear()

      counterBackPropagation=0
      this.X.clear()
      true
    }
    else
      false
  }

  def FeedForwardTest(correlationId: String, startIndex: Int, endIndex: Int, index: Int, data: Array[Float], internalSubLayer: Int, layer: Int):  Array[Array[Float]] = {
    if (!wTest) {
      wTest = true
    }

    val nextLayer = layer + 1
    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      this.XTest += (correlationId -> data)
    }

    val w1 = MatrixHelper.matrixMult(weights, this.XTest(correlationId), Network.getHiddenLayers(nextLayer, "hidden"))
    val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_0")
    actorHiddenLayer ! ComputeActivation.FeedForwardTest(correlationId, w1, 0, internalSubLayer, nextLayer, Network.InputLayerDim)
    weighted -= (correlationId)
    minibatch -= (correlationId)
    XTest -= (correlationId)
    null
  }
}
