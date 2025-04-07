package com.deeplearning.layer

import breeze.linalg.DenseVector
import com.deeplearning.MatrixHelper.{Compute, dotProduct3, dotProduct4, dotProduct6}
import com.deeplearning.Network.heInitialization
import com.deeplearning.{ComputeActivation, ComputeOutput, ComputeWeighted, LayerManager, Network}


class FlattenWeightedLayer extends WeightedLayer {
  var activationsLength:Int = 0

  def Weights(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, activations: Array[Float], Shards:Int, internalSubLayer: Int, layer: Int, params : scala.collection.mutable.HashMap[String,String]) : Array[Array[Float]] = {
    if (this.lastEpoch != epoch) {
      this.counterTraining = 0
      this.counterBackPropagation = 0
      this.counterFeedForward = 0
      this.lastEpoch = epoch
    }

    val nextLayer = layer + 1
    this.counterTraining += 1

    var hiddenLayerStep = 0
    var arraySize = 0

    if (!LayerManager.IsLast(layer)) {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(layer, "weigthed")
      arraySize = Network.getHiddenLayersDim(layer, "weigthed")
    }
    else {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
    }

    if (!this.wInitialized) {
      this.activationsLength = activations.length
      this.weights = Array.ofDim(arraySize)
      this.nabla_w = Array.ofDim(arraySize)
      /*
      for (i <- 0 until arraySize) {
        this.weights(i) = Array.ofDim(hiddenLayerStep)
        this.nabla_w(i) = Array.ofDim(hiddenLayerStep)

        for (j <- 0 until hiddenLayerStep) {
          this.weights(i)(j) = heInitialization(activations.length, 1, Network.getHiddenLayersDim(layer))
          this.nabla_w(i)(j) = Array.fill[Float](activations.length)(0)
        }

      }

       */
      this.wInitialized = true
    }

    this.activation += (correlationId -> activations)
    if (!this.minibatch.contains(correlationId)) {
      this.minibatch += (correlationId -> 0)
      this.messagePropagateReceived += (correlationId -> 0)
      this.fromInternalReceived += (correlationId -> 0)
      this.activation += (correlationId -> activations)

      //context.log.info(s"Randomly initializing weights for layer $layer section $internalSubLayer of length ${inputs.length}")
      var weighted_tmp: Array[Array[Float]] = Array(Array[Float]())
      var nablaw_tmp: Array[Array[Float]] = Array(Array[Float]())
      var nablasw_tmp: Array[Array[Array[Float]]] = Array(Array(Array[Float]()))

      weighted_tmp = Array.ofDim(arraySize)
      nablaw_tmp = Array.ofDim(arraySize)
      nablasw_tmp = Array.ofDim(arraySize)
      for (i <- 0 until arraySize) {
        nablasw_tmp(i) = Array.ofDim(activations.length)
        nablaw_tmp(i) = Array.ofDim(activations.length)
        weighted_tmp(i) = Array.fill[Float](activations.length)(0)
      }

      //this.weighted += (correlationId -> weighted_tmp)
      //this.nablas_w += (correlationId -> nablasw_tmp)
    }

    // create a context and command queue for the GPU
    // create a tensor from the array
    // send to actors for Z compute
    //for (i <- 0 until arraySize) {
     // this.weighted(correlationId)(i) = dotProduct6(LayerManager.GetHiddenLayerStep(layer), this.weights(i), activations)
    //}

    val activation = this.weighted(correlationId)
    val callersize = Network.getHiddenLayersDim(layer, "weigthed")

    if (!LayerManager.IsLast(nextLayer)) {
      for (i: Int <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
        val arr = activation(i)
        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
     //   actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, trainingCount, arr, i, nextLayer, callersize,null)
      }
    }
    else if (LayerManager.IsLast(nextLayer)) {
      for (i <- 0 until Network.OutputLayerDim) {
        val arr = activation(i)
        val actorOutputLayer = Network.LayersOutputRef("outputLayer_" + i)
      //  actorOutputLayer ! ComputeOutput.Compute(epoch, correlationId, yLabel, trainingCount, arr, i, layer, callersize,null)
      }
    }
    activation.grouped(1).toArray
  }

  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int, params : scala.collection.mutable.HashMap[String,String], applyGrads: Boolean) : Boolean = {
    this.counterBackPropagation += 1
    if (layer == 0) {
      val test = 1
    }

    var hiddenLayerStep = 0
    var arraySize = 0

    this.minibatch(correlationId) += 1
    this.messagePropagateReceived(correlationId) += 1
    val fromLayer = layer + 1
    var fromArraySize = 0
    if ((fromLayer-1) == Network.HiddenLayersDim.length) {
      fromArraySize = Network.OutputLayerDim
    }
    else {
      fromArraySize = Network.getHiddenLayersDim(fromLayer, "weigthed")
    }

    if (!backPropagateReceived.contains(correlationId)) {
      backPropagateReceived += (correlationId -> true)
      nablas_w(correlationId) = Array.ofDim(fromArraySize)
    }

 //   nablas_w(correlationId)(fromInternalSubLayer) = dotProduct3(delta.length, delta, activation(correlationId))

    //compute new delta
    val nextlayer = layer + 1
    var callerSize = 0
    if (LayerManager.IsLast(nextlayer)) {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
    }
    else {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(layer, "weigthed")
      arraySize = Network.getHiddenLayersDim(layer, "weigthed")
    }

   // val newdelta = dotProduct4(LayerManager.GetHiddenLayerStep(layer), weights(fromInternalSubLayer), delta)

    callerSize = arraySize * Network.getHiddenLayersDim(layer, "weigthed")
    // only send when all the call from the layer +1 are received
    if (fromArraySize == messagePropagateReceived(correlationId)) {

      fromInternalReceived(correlationId) += 1
      if (Network.getHiddenLayersType(layer-1, "hidden") == "Conv2d") {
        val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + (layer-1) + "_" + internalSubLayer)
     //   hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, newdelta, learningRate, regularisation, nInputs, layer-1, internalSubLayer,null)
      }
      else {
        val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + internalSubLayer)
       // hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, newdelta, learningRate, regularisation, nInputs, layer, internalSubLayer,null)
      }
    }

    // check if we reach the last mini-bacth
    if (Network.MiniBatch * fromArraySize == minibatch.values.sum) {

      //[ (1-eta*(lmbda/n))  *  w  -  (eta/len(mini_batch))  *nw
      //nabla_w = Array.ofDim(arraySize)
      for (i <- 0 until arraySize) {
        nabla_w(i) = Array.tabulate(hiddenLayerStep, this.activationsLength)((_, _) => 0)
        for ((k, w) <- nablas_w) {
          //if (nabla_w(i)==null)
           // nabla_w(i) =  nablas_w(k)(i)
          //else
           // nabla_w(i) = (DenseVector(nabla_w(i)) + DenseVector(nablas_w(k)(i))).toArray

        }
      }

      //mean of all the splits
      /*
        for (i <- 0 until arraySize) {
          this.nabla_w(i) = this.nabla_w(i).map(arr => arr.map(_ / Network.MiniBatch))
        }
         */

      /*
      val dimensions = (arraySize, hiddenLayerStep)
      this.weights = Array.tabulate(dimensions._1, dimensions._2) { (i, j) =>
        val tmp2 = (learningRate / Network.MiniBatch) * DenseVector(nabla_w(i)(j))
        val tmp = this.weights(i)(j)

        val tmp1 = (1 - learningRate * (regularisation / nInputs)) * DenseVector(tmp)
        (tmp1 - tmp2).toArray
      }

       */
      for (i <- 0 until arraySize) {
        for (j <- 0 until hiddenLayerStep) {
      //    val tmp = this.weights(i)(j)
       //   val tmp1 = (1 - learningRate * (regularisation / nInputs)) * DenseVector(tmp)
          val tmp2 = (learningRate / Network.MiniBatch) * DenseVector(nabla_w(i)(j))
       //   weights(i)(j) = (tmp1 - tmp2).toArray
        }
      }

      fromInternalReceived.clear()
      activation.clear()
      backPropagateReceived.clear()
      messagePropagateReceived.clear()
      minibatch.clear()
      nablas_w.clear()
      weighted.clear()
      true
    }
    else
      false
  }

  override def FeedForwardTest(correlationId: String, activations: Array[Float], internalSubLayer: Int, layer: Int): Array[Array[Float]] =  {
    val nextLayer = layer + 1
    counterFeedForward += 1

    var hiddenLayerStep = 0
    var arraySize = 0

    if (LayerManager.IsLast(nextLayer)) {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
    }
    else {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(nextLayer, "hidden")
      arraySize = Network.getHiddenLayersDim(nextLayer, "hidden")
    }

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      activation += (correlationId -> activations)

      //context.log.info(s"Randomly initializing weights for layer $layer section $internalSubLayer of length ${inputs.length}")
      var weighted_tmp: Array[Array[Float]] = Array(Array[Float]())
      weighted_tmp = Array.ofDim(arraySize)
      for (i <- 0 until arraySize) {
        weighted_tmp(i) = Array.fill[Float](hiddenLayerStep)(0)
      }
    //  weighted += (correlationId -> weighted_tmp)
    }

    // send to actors for Z compute
    if (!LayerManager.IsLast(nextLayer)) {
      for (i <- 0 until arraySize) {
      //  this.weighted(correlationId)(i) = dotProduct6(LayerManager.GetHiddenLayerStep(layer), this.weights(i), activations)
      }
    }
    else {
      for (i <- 0 until Network.OutputLayerDim) {
        //this.weighted(correlationId)(i) = dotProduct6(LayerManager.GetHiddenLayerStep(layer), this.weights(i), activations)
      }
    }

    val array = this.weighted(correlationId)

    if (!LayerManager.IsLast(nextLayer)) {
      for (i: Int <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
        val arr = this.weighted(correlationId)(i)
        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
        //actorHiddenLayer ! ComputeActivation.FeedForwardTest( correlationId, arr, i, nextLayer, 1)
      }
    }
    else if (LayerManager.IsLast(nextLayer)) {
      for (i <- 0 until Network.OutputLayerDim) {
        val arr = this.weighted(correlationId)(i)
        val actorOutputLayer = Network.LayersOutputRef("outputLayer_" + i)
       // actorOutputLayer ! ComputeOutput.FeedForwardTest( correlationId, arr, i, layer, 1)
      }
    }


    activation -= correlationId
    weighted -= correlationId
    minibatch -= correlationId

    array.grouped(1).toArray
  }

  override def getNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int, localIndex:Int): Unit = {
    weighted(correlationId)
  }

  def SynchronizeWeights(correlationId:String, gradients:Array[Float], layer:Int, internalSubLayer:Int): Unit = {

  }

  override def sendGradients(correlationId: String, gradients: Array[Float], layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int): Unit = {

  }

  override def applyGradients(correlationId: String, gradients: Array[Float], layer: Int,  internalSubLayer:Int, fromInternalSubLayer: Int): Unit = {

  }
}
