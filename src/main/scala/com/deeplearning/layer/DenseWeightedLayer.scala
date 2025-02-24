package com.deeplearning.layer


import ai.djl.Device
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.{NDArray, NDManager}
import com.deeplearning.CostManager.{getIndex, matMul2, roundUpIfFractional}
import com.deeplearning.Network.generateRandomFloat
import com.deeplearning.{ComputeActivation, ComputeOutput, ComputeWeighted, CostManager, LayerManager, Network}

import scala.math._
import java.time.{Duration, Instant}


class DenseWeightedLayer extends WeightedLayer {
  var activationsLength:Int = 0
  var parameterSended = false
  val parameters = scala.collection.mutable.HashMap.empty[String, String]
  val deltas2 = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  val nablas_w_tmp2 = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  val shardReceived = scala.collection.mutable.HashMap.empty[String, Int]
  val inProgress = scala.collection.mutable.HashMap.empty[String, Boolean]
  val syncReceived = scala.collection.mutable.HashMap.empty[String, Array[Int]]
  val gradientsSync = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  val callerBuffered = scala.collection.mutable.HashMap.empty[String, Array[Array[Int]]]

  private var ucIndex = 0
  def Weights(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, activations: Array[Float], shards: Int, internalSubLayer: Int, layer: Int, params : scala.collection.mutable.HashMap[String,String]) : Array[Array[Float]] = {
    if (this.lastEpoch != epoch) {
      this.counterTraining = 0
      this.counterBackPropagation = 0
      this.counterFeedForward = 0
      this.lastEpoch = epoch
      this.inProgress.clear()
      this.shardReceived.clear()
      this.syncReceived.clear()
      this.callerBuffered.clear()
      this.gradientsSync.clear()
    }

    val startedAt = Instant.now
    val nextLayer = layer + 1
    this.counterTraining += 1

    var hiddenLayerStep = 0
    var arraySize = 0
    var nextLayerSize = 0
    if (!LayerManager.IsLast(nextLayer)) {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(nextLayer)
      arraySize = 1
      nextLayerSize = Network.getHiddenLayers(nextLayer, "hidden")
    }
    else {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
      nextLayerSize = Network.OutputLayer
    }

    var split = 0
    if (!LayerManager.IsLast(nextLayer))
      split = Network.getHiddenLayers(nextLayer, "hidden")
    else
      split = Network.OutputLayer
    val sizeact = Network.getHiddenLayersDim(layer, "weighted")
    if (!this.wInitialized) {
      ucIndex = internalSubLayer
      this.activationsLength = activations.length

      if (sizeact>1) {
        val tmp = generateRandomFloat(nextLayerSize*activationsLength).grouped(activationsLength*nextLayerSize /sizeact).toArray
        this.weights =  tmp(internalSubLayer)
      }
      else {
        this.weights = generateRandomFloat(split*activationsLength)
      }

      this.wInitialized = true
      //parameters += ("min" -> "0")
      //parameters += ("max" -> "0")
      //parameters += ("weighted_min" -> "0")
      //parameters += ("weighted_max" -> "0")
    }

    //this.activation += (correlationId -> activations)
    if (!this.minibatch.contains(correlationId)) {
      this.gradientsSync += (correlationId -> Array.ofDim(1))
      this.minibatch += (correlationId -> 0)
      this.messagePropagateReceived += (correlationId -> 0)
      this.fromInternalReceived += (correlationId -> 0)
      this.activation += (correlationId -> Array.fill[Float](activations.size)(0.0f))
      this.syncReceived += (correlationId -> Array.ofDim(1))
      this.callerBuffered += (correlationId -> Array.ofDim(1))
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
    }
    shardReceived(correlationId) +=1

    if (shardReceived(correlationId) < shards && inProgress(correlationId)) {
      activation(correlationId) = CostManager.sum2(activation(correlationId), activations)
      null
    }
    else {
      val fromUCs = Network.getHiddenLayersDim(layer, "weighted")
      activation(correlationId) = activations

      inProgress(correlationId) = false
      var w1 = Array.fill(split)(0.0f)
      if (sizeact>1) {
        val weigthsGrouped = weights.grouped(activationsLength).toArray
        val dim = weigthsGrouped.length
        val residu1 = if (weigthsGrouped(dim-1).size == activationsLength) 0 else weigthsGrouped(0).size- weigthsGrouped(dim-1).size
        val residu = activationsLength - residu1

        if (residu1 == 0 ) {
          val weigthsGrouped = weights.grouped(activationsLength).toArray
          w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, dim)
        }
        else if (internalSubLayer == 0 ) {
          val weigthsGrouped = weights.grouped(activationsLength).toArray
          weigthsGrouped(weigthsGrouped.size-1) = weigthsGrouped(weigthsGrouped.size-1).padTo(activationsLength, 0.0f)
          w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, dim)
        }
        else if ( (internalSubLayer+1) < sizeact) {
          val indexes = CostManager.getRange(weights.length, activationsLength, nextLayerSize, internalSubLayer)
          val weightstmp = Array.fill(indexes(0))(0.0f) ++ weights ++  Array.fill(indexes(1))(0.0f)
          val weigthsGrouped = weightstmp.grouped(activationsLength).toArray
          w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, weigthsGrouped.length)
        }
        else if ((internalSubLayer+1) ==sizeact) {
          val weightstmp = Array.fill(residu1)(0.0f) ++ weights
          val weigthsGrouped = weightstmp.grouped(activationsLength).toArray
          w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, dim)
        }

      }
      else {
        val a = activation(correlationId)
        w1 = CostManager.initInputs(weights, activation(correlationId), split)
      }

      weighted(correlationId) = w1

      if (Network.CheckNaN) {
        val nanIndices = weighted(correlationId).zipWithIndex.filter { case (value, _) => value.isNaN || value == 0f }
        // Check if there are any NaN values
        if (nanIndices.nonEmpty) {
          println("NaN values found at indices:")
          nanIndices.foreach { case (_, index) => println(index) }
        } else {
          println("No NaN values found in the array.")
        }
      }

      val endedAt = Instant.now
      val duration = Duration.between(startedAt, endedAt).toMillis

      if (counterTraining % Network.minibatchBuffer == 0 && Network.debug) {
        println("-------------------------------------------")
        println("Weighted init duration : " + duration)
      }

      if (!parameterSended) {
        //    parameters("min") = weights.min.toString
        //    parameters("max") = weights.max.toString
        parameterSended = true
      }

      if (!LayerManager.IsLast(nextLayer)) {
        val toNeurons = Network.getHiddenLayers(nextLayer,"hidden")
        val toUCs = Network.getHiddenLayersDim(nextLayer, "hidden")
        var splitWeights = Array(w1)

        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_0")
        actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, trainingCount, w1, 0, internalSubLayer, nextLayer, fromUCs,params, weights)

      }
      else if (LayerManager.IsLast(nextLayer)) {
        for (i <- 0 until Network.OutputLayerDim) {
          val actorOutputLayer = Network.LayersOutputRef("outputLayer_" + i)
          actorOutputLayer ! ComputeOutput.Compute(epoch, correlationId, yLabel, trainingCount, w1, i, internalSubLayer, layer+1, fromUCs, params)
        }
      }
      null
    }
  }

  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int, params : scala.collection.mutable.HashMap[String,String], applyGrads:Boolean) : Boolean = {
    var hiddenLayerStep = 0
    var arraySize = 0

    val fromLayer = layer + 1
    var fromArraySize = 0
    val nextlayer = layer + 1

    if ((fromLayer - 1) == Network.HiddenLayersDim.length * 2) {
      fromArraySize = Network.OutputLayerDim
    }
    else {
      fromArraySize = Network.getHiddenLayersDim(nextlayer, "hidden")
    }

      this.minibatch(correlationId) += 1
      this.messagePropagateReceived(correlationId) += 1
      this.counterBackPropagation += 1

      if (!backPropagateReceived.contains(correlationId)) {
        backPropagateReceived += (correlationId -> true)
      }

      var callerSize = 0
      var group = 0
      if (LayerManager.IsLast(nextlayer)) {
        hiddenLayerStep = LayerManager.GetOutputLayerStep()
        arraySize = Network.OutputLayerDim
        group = LayerManager.GetDenseWeightedLayerStep(layer)
      }
      else if (LayerManager.IsFirst(layer - 1)) {
        arraySize = Network.InputLayerDim
        group = Network.InputLayerDim
      }
      else {
        arraySize = Network.getHiddenLayersDim(layer, "weighted")
        group = LayerManager.GetDenseWeightedLayerStep(layer)
      }

      val startedAt = Instant.now
      nablas_w_tmp2 += (correlationId ->  activation(correlationId))
      deltas2 += (correlationId -> delta)
      var endedAt = Instant.now
      var duration = Duration.between(startedAt, endedAt).toMillis

      val sectionSize = Network.getHiddenLayersDim(layer, "hidden")
      val layerSize = Network.getHiddenLayers(layer, "hidden")
      val sizeact = Network.getHiddenLayersDim(layer, "hidden")

      if (sizeact > 1) {
        val weigthsGrouped = weights.grouped(activationsLength).toArray
        val dim = weigthsGrouped.size

        var w1 = Array.fill(1)(0.0f)
        val sliceSize = delta.size * activationsLength / sizeact
        val f = roundUpIfFractional(sliceSize/activationsLength.toDouble)
        var startIndex = 0
        var endIndex = 0

        val residu1 = if (weigthsGrouped(dim-1).size == activationsLength) 0 else weigthsGrouped(0).size- weigthsGrouped(dim-1).size
        val residu = activationsLength - residu1

        if (residu1 == 0 ) {
          if (internalSubLayer == 0) {
            val sizedlt = delta.length/sizeact
            val deltatmp = delta.take(sizedlt)
            w1 = CostManager.dotInputs(weights,deltatmp, activationsLength)
          }
          else if ((internalSubLayer+1) < sizeact) {
            val sizedlt = delta.length/sizeact
            val deltatmp = delta.slice(sizedlt*internalSubLayer, sizedlt*internalSubLayer+sizedlt)
            w1 = CostManager.dotInputs(weights,deltatmp, activationsLength)
          }
          else {
            val sizedlt = delta.length/sizeact
            val deltatmp = delta.slice(sizedlt*internalSubLayer, delta.length)
            w1 = CostManager.dotInputs(weights,deltatmp, activationsLength)
          }
        }
        else if (internalSubLayer==0) {
          val weightstmp = weights.grouped(activationsLength).toArray
          weigthsGrouped(weightstmp.size-1) = weightstmp(weightstmp.size-1).padTo(activationsLength, 0.0f)
          startIndex = f*internalSubLayer-1*internalSubLayer
          endIndex = startIndex+f
          val dlt2 = delta.slice(startIndex, endIndex)
          w1 = CostManager.dotInputs( weigthsGrouped.flatten,dlt2, activationsLength)
        }
        else if ((internalSubLayer+1) < sizeact) {
          val indexes = CostManager.getRange(weights.length, activationsLength, fromArraySize, internalSubLayer)
          var weightstmp = Array.fill(indexes(0))(0.0f) ++ weights ++  Array.fill(indexes(1))(0.0f)
          /*
          if (internalSubLayer == 1  || internalSubLayer == 4) {
            if (sizeact == 3)
              weightstmp = Array.fill(residu)(0.0f) ++ weights ++  Array.fill(residu)(0.0f)
            else
              weightstmp = Array.fill(60)(0.0f) ++ weights ++  Array.fill(60)(0.0f)
          }
          else if (internalSubLayer == 2 || internalSubLayer == 5) {
            weightstmp = Array.fill(30)(0.0f) ++ weights
          }
          else if (internalSubLayer == 3 ) {
            weightstmp = weights ++  Array.fill(30)(0.0f)
          }
          */
          var offset = 0
          val multiplier = if (delta.length%sectionSize==0)delta.length/sectionSize else delta.length/sectionSize+1

          startIndex = f*internalSubLayer-1*internalSubLayer

          if (internalSubLayer == 4)
            offset = -1
          startIndex = multiplier*internalSubLayer-1+offset
          startIndex = getIndex(sizeact, delta.length, internalSubLayer)
          val weigthsGrouped = weightstmp.grouped(activationsLength).toArray
          endIndex = startIndex+weigthsGrouped.length
          val dlt2 = delta.slice(startIndex, endIndex)
          w1 = CostManager.dotInputs( weigthsGrouped.flatten,dlt2, activationsLength)
        }
        else if ((internalSubLayer+1) ==sizeact) {

          val weightstmp = Array.fill(residu1)(0.0f) ++ weights
          val weigthsGrouped = weightstmp.grouped(activationsLength).toArray

          val dlt = delta.slice(delta.length - weigthsGrouped.length, delta.length)
          w1 = CostManager.dotInputs( weigthsGrouped.flatten,dlt, activationsLength)
        }
        if (layer ==2) {
          val test = 1
        }
        backPropagation(correlationId, w1, learningRate, regularisation, nInputs, sizeact, layer, params)
      }
      else {
        if (layer ==2) {
          val test = 1
        }
        val w1 = CostManager.dotInputs(weights,delta, group)
        backPropagation(correlationId, w1, learningRate, regularisation, nInputs, sizeact, layer, params)
      }

      if (Network.MiniBatch   == minibatch.values.sum) {
        parameterSended = false
        parameters("min") = "0"
        parameters("max") = "0"

        var fromUCs = 1
        if (!LayerManager.IsLast(nextlayer))
          fromUCs = Network.getHiddenLayersDim(layer + 1, "hidden")

        endedAt = Instant.now
        duration = Duration.between(startedAt, endedAt).toMillis

        if (sizeact>1) {
          // Iterating over the HashMap

          var w2:Array[Float] = null
          val act = nablas_w_tmp2.values.flatten.toArray
          val deltas = deltas2.values.flatten.toArray
          val cpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
          val array1 = cpuManager.create(deltas)
          val array2 = cpuManager.create(act)
          val fromMat1: NDArray = cpuManager.from(array1)
          val fromMat2: NDArray = cpuManager.from(array2)
          // 3. Perform linear transformation

          // Perform the outer product
          val mat1Reshaped = fromMat1.reshape(Network.MiniBatch,deltas.size/ Network.MiniBatch, 1)
          val mat2Reshaped = fromMat2.reshape(Network.MiniBatch, 1, act.size/Network.MiniBatch)

          val result = mat1Reshaped.matMul(mat2Reshaped)
          val tr = result.toFloatArray
          var s = result.sum(Array(0)).reshape(-1).toFloatArray

          // Reshape the result to [50,1]

          if (internalSubLayer==0) {
            s = s.take(weights.length)
          }
          else if (internalSubLayer+1<sizeact) {
            s = s.slice(weights.length*internalSubLayer, weights.length*internalSubLayer+weights.length)
          }
          else {
            s = s.slice(weights.length*internalSubLayer, s.length)
          }

          fromMat2.close()
          fromMat1.close()
          cpuManager.close()
          if (w2==null)
            w2 = s
          else
            w2 = CostManager.sum2(w2,s)


          this.weights = CostManager.applyGradientsLight(w2, weights,Network.MiniBatch,learningRate / Network.MiniBatch, 1 - learningRate * (regularisation / nInputs))
        }
        else {
          val act2 = nablas_w_tmp2.values.flatten.toArray
          val del2 = deltas2.values.flatten.toArray
          this.weights = CostManager.applyGradients(del2, act2, Network.MiniBatch, learningRate / Network.MiniBatch, 1 - learningRate * (regularisation / nInputs), this.weights)
        }
        gradientsSync.clear()
        nablas_w_tmp2.clear()
        fromInternalReceived.clear()
        activation.clear()
        backPropagateReceived.clear()
        messagePropagateReceived.clear()
        minibatch.clear()
        nablas_w.clear()
        weighted.clear()
        deltas2.clear()
        shardReceived.clear()
        inProgress.clear()

      }
    true
  }

  def backPropagation(correlationId :String, delta: Array[Float], learningRate : Float, regularisation:Float, nInputs:Int, shards:Int, layer:Int, params : scala.collection.mutable.HashMap[String,String]) : Unit = {
    if (layer > 1 && Network.getHiddenLayersType(layer - 1, "hidden") == "Conv2d") {
      val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + (layer - 1) + "_" + ucIndex)
      hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, delta, learningRate, regularisation, nInputs, shards, layer - 1, ucIndex,ucIndex, params)
    }
    else {
      val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + (layer - 1) + "_0")
      if (Network.debugActivity)
        println("Send AL " + (layer - 1) + " " + ucIndex)

      hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId,delta, learningRate, regularisation, nInputs, shards, layer - 1, ucIndex,ucIndex, params)
    }
  }


  override def sendGradients(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int, fromInternalSubLayer:Int): Unit = {
    val gr = gradientsSync(correlationId)(fromInternalSubLayer)
    val actorHiddenLayer = Network.LayersIntermediateRef("weighted_" + layer + "_" + internalSubLayer)
    actorHiddenLayer ! ComputeWeighted.getGradientsNeighbor(correlationId, gr, layer,internalSubLayer,fromInternalSubLayer)
  }

  override def applyGradients(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int,fromInternalSubLayer: Int): Unit = {
    syncReceived(correlationId)(fromInternalSubLayer) += 1
    if (gradientsSync(correlationId)(fromInternalSubLayer) == null)
      gradientsSync(correlationId)(fromInternalSubLayer) = gradients
    else
      gradientsSync(correlationId)(fromInternalSubLayer) = CostManager.sum2(gradientsSync(correlationId)(fromInternalSubLayer),gradients)

    val currentLayerSize = Network.getHiddenLayersDim(layer, "weightedLayer")
   // println(syncReceived(correlationId) + "  layer : " + layer + " : " + internalSubLayer + " "  + fromInternalSubLayer)
    if ((syncReceived(correlationId)(fromInternalSubLayer)+1) == currentLayerSize) {
//      println(correlationId + " backpropagation done Layer : " + layer + " : " + internalSubLayer + " "  + fromInternalSubLayer)
      backPropagation(correlationId,gradientsSync(correlationId)(fromInternalSubLayer),Network.LearningRate, Network.Regularisation, Network.MiniBatchRange,fromInternalSubLayer, layer,scala.collection.mutable.HashMap.empty[String, String])
    }

  }

  def SynchronizeWeights(correlationId:String, gradients:Array[Float], layer:Int, internalSubLayer:Int, fromInternalSubLayer: Int): Unit = {
    // check if a buffer is waiting
    if (callerBuffered(correlationId)(fromInternalSubLayer) != null) {
      // Iterate over the hashTable
      val test = callerBuffered(correlationId)(fromInternalSubLayer)
      for (array <- callerBuffered(correlationId)(fromInternalSubLayer)) {
        // Iterate over each (x, y) pair in the array
        val i = array
        val actorWeightedLayer = Network.LayersIntermediateRef("weightedLayer_" + layer + "_" + i)
        val gradient = gradientsSync(correlationId)(fromInternalSubLayer)
     //   println("----------------------------------------------------------------------------------")
     //   println("buffered " + ucIndex + " : "  + correlationId + " " + layer + " " + i + " " + fromInternalSubLayer)
     //   println("----------------------------------------------------------------------------------")
        actorWeightedLayer ! ComputeWeighted.fromGradientsNeighbor(correlationId, gradient, layer, i, fromInternalSubLayer)
      }
      callerBuffered -= (correlationId)
    }
    val x = Network.getHiddenLayersDim(layer, "weighted")
    for (i <- 0 until x if i != internalSubLayer) {
      val actorHiddenLayer = Network.LayersIntermediateRef("weightedLayer_" + layer + "_" + i)
      actorHiddenLayer ! ComputeWeighted.getGradientsNeighbor(correlationId, gradients, layer, internalSubLayer, fromInternalSubLayer)
    }
  }

  override def getNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, fromInternalSubLayer: Int) : Unit = {
    val c = gradientsSync(correlationId)(fromInternalSubLayer)
    // gradients not processed yet
    if (c==null) {
      if (callerBuffered(correlationId)(fromInternalSubLayer) ==null)
        callerBuffered(correlationId)(fromInternalSubLayer) = Array(internalSubLayerCaller)
      else {
        callerBuffered(correlationId)(fromInternalSubLayer)  :+= internalSubLayerCaller
      }
    }
    else {
      val actorHiddenLayer = Network.LayersIntermediateRef("weightedLayer_" + layer + "_" + internalSubLayerCaller)
      actorHiddenLayer ! ComputeWeighted.fromGradientsNeighbor(correlationId, c, layer, internalSubLayerCaller, fromInternalSubLayer)
    }
  }
  override def FeedForwardTest(correlationId: String, activations: Array[Float], ucIndex: Int, layer: Int): Array[Array[Float]] =  {
    val nextLayer = layer + 1
    counterFeedForward += 1

    var hiddenLayerStep = 0
    var arraySize = 0

    if (!LayerManager.IsLast(nextLayer)) {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(nextLayer)
      arraySize = Network.getHiddenLayersDim(nextLayer, "hidden")
    }
    else {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
    }
    var split = 0
    if (!LayerManager.IsLast(nextLayer))
      split = Network.getHiddenLayers(nextLayer, "hidden")
    else
      split = Network.OutputLayer

    inProgress(correlationId) = false
    var w1 = Array.fill(split)(0.0f)
    val sizeact = Network.getHiddenLayersDim(layer, "hidden")
    activation(correlationId) = activations

    if (sizeact>1) {
      val weigthsGrouped = weights.grouped(activationsLength).toArray
      val dim = weigthsGrouped.length
      val residu1 = if (weigthsGrouped(dim-1).size == activationsLength) 0 else weigthsGrouped(0).size- weigthsGrouped(dim-1).size
      val residu = activationsLength - residu1

      if (residu1 == 0 ) {
        val weigthsGrouped = weights.grouped(activationsLength).toArray
        w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, dim)
      }
      else if (ucIndex == 0 ) {
        val weigthsGrouped = weights.grouped(activationsLength).toArray
        weigthsGrouped(weigthsGrouped.size-1) = weigthsGrouped(weigthsGrouped.size-1).padTo(activationsLength, 0.0f)
        w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, dim)
      }
      else if ( (ucIndex+1) < sizeact) {
        val indexes = CostManager.getRange(weights.length, activationsLength, layer+1, ucIndex)
        val weightstmp = Array.fill(indexes(0))(0.0f) ++ weights ++  Array.fill(indexes(1))(0.0f)
        val weigthsGrouped = weightstmp.grouped(activationsLength).toArray
        w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, weigthsGrouped.length)
      }
      else if ((ucIndex+1) ==sizeact) {
        val weightstmp = Array.fill(residu1)(0.0f) ++ weights
        val weigthsGrouped = weightstmp.grouped(activationsLength).toArray
        w1 =  CostManager.initInputs(weigthsGrouped.flatten, activations, dim)
      }


    }
    else {
      val a = activation(correlationId)
      w1 = CostManager.initInputs(weights, activation(correlationId), split)
    }

    weighted(correlationId) = w1

      if (Network.CheckNaN) {
        val nanIndices = weighted(correlationId).zipWithIndex.filter { case (value, _) => value.isNaN || value == 0f }
        // Check if there are any NaN values
        if (nanIndices.nonEmpty) {
          println("NaN values found at indices:")
          nanIndices.foreach { case (_, index) => println(index) }
        } else {
          println("No NaN values found in the array.")
        }
      }

      if (!LayerManager.IsLast(nextLayer)) {
        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_0")
        actorHiddenLayer ! ComputeActivation.FeedForwardTest(correlationId,w1, 0, ucIndex, nextLayer, sizeact)
      }
      else if (LayerManager.IsLast(nextLayer)) {
        for (i <- 0 until Network.OutputLayerDim) {
          val actorOutputLayer = Network.LayersOutputRef("outputLayer_" + i)
          actorOutputLayer ! ComputeOutput.FeedForwardTest(correlationId, w1, i, ucIndex, layer+1, sizeact)
        }
      }

    activation -= correlationId
    weighted -= correlationId
    minibatch -= correlationId

    null
  }

}
