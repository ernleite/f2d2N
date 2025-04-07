package com.deeplearning.layer

import com.deeplearning.MatrixHelper.{dotProduct, getIndex}
import com.deeplearning.{ActivationManager, ComputeActivation, ComputeInputs, ComputeWeighted, MatrixHelper, LayerManager, Network, Normalisation}
import com.deeplearning.Network.generateRandomFloat

class DenseActivationLayer extends ActivationLayer {
  private val parameters = scala.collection.mutable.HashMap.empty[String,String]
  private val weightedMin = scala.collection.mutable.HashMap.empty[String,Float]
  private val weightedMax = scala.collection.mutable.HashMap.empty[String, Float]
  private val deltaTmp = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private val weightsTmp = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private var gamma = 0.1f
  private var beta = 0.1f
  private var ucIndex = 0
  val syncReceived = scala.collection.mutable.HashMap.empty[String, Array[Int]]
  val gradientsSync = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  val callerBuffered = scala.collection.mutable.HashMap.empty[String, Array[Int]]
  val bpShardReceived = scala.collection.mutable.HashMap.empty[String, Int]
  val deltaSync = scala.collection.mutable.HashMap.empty[String, Array[Float]]


  override def ComputeZ(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, shardedWeighted: Array[Float], internalSubLayer: Int, fromUCIndex:Int, layer: Int, shards: Int, params : scala.collection.mutable.HashMap[String,String]): Array[Float] = {
    if (lastEpoch != epoch) {
      counterTraining = 0
      counterBackPropagation = 0
      counterFeedForward = 0
      lastEpoch = epoch
      this.shardReceived.clear()
      this.bpShardReceived.clear()
      this.syncReceived.clear()
      this.callerBuffered.clear()
      this.gradientsSync.clear()
      this.deltaSync.clear()
    }
    // bias initialized only one time during the training cycle
    counterTraining += 1
    if (!bInitialized) {
      //bias = generateRandomBiasFloat(LayerManager.GetDenseActivationLayerStep(layer))
      bias = generateRandomFloat(LayerManager.GetDenseActivationLayerStep(layer))
      bInitialized = true
      parameters += ("min" -> "0")
      parameters += ("max" -> "0")
      ucIndex = internalSubLayer
    }
    val activationLength = bias.length
    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      messagePropagateReceived += (correlationId -> 0)
      weightedMin += (correlationId -> 0f)
      weightedMax += (correlationId -> 0f)
      this.syncReceived += (correlationId -> Array.ofDim(1))
      this.callerBuffered += (correlationId -> null)
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
      bpShardReceived += (correlationId -> 0)
      weighted += (correlationId -> Array.fill[Float](activationLength)(0.0f))
    }

    // shards : number of vertical UCs from the previous WL layer
    // correlationId = Sample Unique UID
    // sum shardedWeighted until receiving all the shards
    // shardReceived(correlationId) : counter of received message for a specific sample
    if (shardReceived(correlationId) < shards) {
      shardReceived(correlationId) += 1
      // No vertical parallelism
      if (shardedWeighted.size == activationLength) {
          weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), shardedWeighted)
      }
      else {
        val biasLength = bias.length
        // First vertical UC 0
        if (fromUCIndex == 0) {
          val weightedTmp = shardedWeighted.padTo(biasLength, 0.0f)
          weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), weightedTmp)
        }
        // from vertical UC 1 to UC -1
        else if ((fromUCIndex + 1) < shards) {
          val index = getIndex(shards, biasLength, fromUCIndex)
          val weightedTmp = Array.fill(biasLength)(0.0f)
          Array.copy(shardedWeighted, 0, weightedTmp, index, shardedWeighted.length)
          weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), weightedTmp)
        }
        // Last vertical UC
        else if ((fromUCIndex + 1) == shards) {
          val act2 = Array.fill(biasLength - shardedWeighted.length)(0.0f) ++ shardedWeighted
          weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), act2)
        }
      }
    }

    if (shards == shardReceived(correlationId) && inProgress(correlationId)) {
      val z = MatrixHelper.matrixSum(weighted(correlationId) , bias)
      Z += (correlationId -> z)
      activation(correlationId)  = MatrixHelper.ComputeZ(Network.getActivationLayersType(layer), z)

      if (Network.dropout > 0) {
        activation(correlationId) = Network.dropout(activation(correlationId))
      }

      if (Network.CheckNaN) {
        val nanIndices = activation(correlationId).zipWithIndex.filter { case (value, _) => value.isNaN || value == 0f }
        // Check if there are any NaN values
        if (nanIndices.nonEmpty) {
          println("NaN values found at indices:")
          nanIndices.foreach { case (_, index) => println(index) }
        } else {
          println("No NaN values found in the array.")
        }
      }

      if (Network.NaN) {
        activation(correlationId) = MatrixHelper.EliminateNaN(activation(correlationId))
      }

      if (Network.LayerNorm) activation(correlationId) = MatrixHelper.layerNorm(activation(correlationId))
      inProgress(correlationId) = false
      shardReceived(correlationId) = 0

      val toUCs = Network.getHiddenLayersDim(layer+1, "weighted")
      if (layer == 1) {
        val test = 1
      }

      for (i <- 0 until toUCs) {
        val actorweightedLayer = Network.LayersIntermediateRef("weightedLayer_" + (layer+1) + "_" + i)
        actorweightedLayer ! ComputeWeighted.Weights(epoch, correlationId, yLabel, trainingCount,  activation(correlationId),1, i, layer+1,params)
      }

      activation(correlationId)
    }
    else
      null
  }

  override def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, shards:Int, layer: Int, internalSubLayer: Int, fromUCIndex:Int, params : scala.collection.mutable.HashMap[String,String]): Boolean = {
    counterBackPropagation += 1
    //compute the derivative
    //context.log.info(s"Receiving backprogation request correlationId $correlationId HiddenLayer_${layer}_${internalSubLayer}")
    messagePropagateReceived(correlationId) += 1
    val verticalParallelism = Network.getHiddenLayersDim(layer+1, "weighted")
    bpShardReceived(correlationId) +=1
    val biasLength = this.bias.length

    if (!backPropagateReceived.contains(correlationId)) {
      backPropagateReceived += (correlationId -> true)
      deltaSync += (correlationId -> Array.fill[Float](biasLength)(0.0f))
    }
    var hiddenLayerStep = 0
    var fromArraySize = 0
    val nextlayer = layer + 1

    if (LayerManager.IsLast(nextlayer)) {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      fromArraySize = Network.OutputLayerDim
    }
    else {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(layer)
      fromArraySize = Network.getHiddenLayersDim(layer, "hidden")
    }

    val multiplier = if (biasLength%shards==0) biasLength / shards
                     else biasLength/shards+1

    if (bpShardReceived(correlationId) <= verticalParallelism) {
      if (biasLength == delta.length) {
        deltaSync(correlationId) = MatrixHelper.matrixSum(deltaSync(correlationId), delta)
      }
      else {
        if (fromUCIndex == 0) {
          val act = delta.padTo(biasLength, 0.0f)
          deltaSync(correlationId) = MatrixHelper.matrixSum(deltaSync(correlationId), act)
        }
        else if ((fromUCIndex+1) < shards) {
          val test = Array.fill(biasLength)(0.0f)
          if (biasLength%delta.length== 0) {
            val index = multiplier*fromUCIndex
            Array.copy(delta, 0, test, index, delta.length)
          }
          else {
            val index = multiplier*fromUCIndex-1
            Array.copy(delta, 0, test, index, delta.length)
          }
          deltaSync(correlationId) = MatrixHelper.matrixSum(deltaSync(correlationId), test)
        }
        else if ( (fromUCIndex+1) == shards) {
          val act2 =  Array.fill(biasLength-delta.length)(0.0f) ++ delta
          deltaSync(correlationId) = MatrixHelper.matrixSum(deltaSync(correlationId), act2)
        }
      }
    }

    if (bpShardReceived(correlationId) == verticalParallelism) {
      val grouped =  deltaSync(correlationId).grouped(activation(correlationId).size).toArray
      val deltaAssembled = MatrixHelper.flattenSum(grouped)
      val prime = MatrixHelper.ComputePrime(Network.getActivationLayersType(layer), this.Z(correlationId))
      minibatch(correlationId) += 1
      val dot = dotProduct(prime, deltaAssembled)
      nablas_b += (correlationId -> dot)
      backPropagation(correlationId,dot,learningRate, regularisation,nInputs, layer,params)
    }

    if (Network.MiniBatch == minibatch.values.sum) {
      val flatten = nablas_b.values.toArray.flatten
      val (tmp4) = MatrixHelper.applyBias(activation(correlationId).size,learningRate / Network.MiniBatch, flatten, bias)
      bias = tmp4

      parameters("min") = "0"
      parameters("max") = "0"

      //if (Network.debug && Network.debugLevel == 4) context.log.info(s"Applying gradients to bias layer $layer / $internalSubLayer")
      nablas_b.clear()
      backPropagateReceived.clear()
      weighted.clear()
      activation.clear()
      inProgress.clear()
      shardReceived.clear()
      minibatch.clear()
      Z.clear()
      weightedMin.clear()
      weightedMax.clear()
      deltaTmp.clear()
      weightsTmp.clear()
      deltaSync.clear()
      bpShardReceived.clear()
      syncReceived.clear()
      true
    }
    else
      false
  }

  override def sendGradients(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int, fromUCIndex:Int): Unit = {
    val gr = gradientsSync(correlationId)
    val actorHiddenLayer = Network.LayersIntermediateRef("weighted_" + layer + "_" + internalSubLayer)
    actorHiddenLayer ! ComputeWeighted.getGradientsNeighbor(correlationId, gr, layer,internalSubLayer,fromUCIndex)
  }

  override def applyGradients(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int,fromUCIndex: Int): Unit = {
    syncReceived(correlationId)(fromUCIndex) += 1
    if (gradientsSync(correlationId)(fromUCIndex) == null)
      gradientsSync(correlationId) = gradients
    else
      gradientsSync(correlationId) = MatrixHelper.matrixSum(gradientsSync(correlationId),gradients)

    val currentLayerSize = Network.getHiddenLayersDim(layer, "weightedLayer")
    // println(syncReceived(correlationId) + "  layer : " + layer + " : " + internalSubLayer + " "  + fromUCIndex)
    if ((syncReceived(correlationId)(fromUCIndex)+1) == currentLayerSize) {
      //      println(correlationId + " backpropagation done Layer : " + layer + " : " + internalSubLayer + " "  + fromUCIndex)
      backPropagation(correlationId,gradientsSync(correlationId),Network.LearningRate, Network.Regularisation, Network.MiniBatchRange,layer,scala.collection.mutable.HashMap.empty[String, String])
    }

  }

  def backPropagation(correlationId :String, delta: Array[Float], learningRate : Float, regularisation:Float, nInputs:Int, layer:Int, params : scala.collection.mutable.HashMap[String,String]) : Unit = {
    val previousLayer = layer - 1
    var hiddenLayerStep = 0
    var fromArraySize = 0

    val nextlayer = layer + 1
    var arraySize = 0

    if (LayerManager.IsLast(nextlayer)) {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      fromArraySize = Network.OutputLayerDim
    }
    else {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(layer)
      fromArraySize = Network.getHiddenLayersDim(layer, "hidden")
    }

    if (LayerManager.IsFirst(previousLayer)) {
      hiddenLayerStep = LayerManager.GetInputLayerStep()
      arraySize = Network.InputLayerDim
      for (i <- 0 until arraySize) {
        val inputLayerRef = Network.LayersInputRef("inputLayer_" + i)
        inputLayerRef ! ComputeInputs.BackPropagate(correlationId, delta, learningRate, regularisation, nInputs, i, ucIndex, params)
      }
    }
    else {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(previousLayer)
      arraySize = Network.getHiddenLayersDim(previousLayer, "hidden")
      for (i <- 0 until arraySize) {
        val intermediateLayerRef = Network.LayersIntermediateRef("weightedLayer_" + previousLayer + "_" + i)
        intermediateLayerRef ! ComputeWeighted.BackPropagate(correlationId, delta, learningRate, regularisation, nInputs, previousLayer, i, ucIndex, params)
      }
    }
  }


  def SynchronizeWeights(correlationId:String, gradients:Array[Float], layer:Int, internalSubLayer:Int, fromUCIndex: Int): Unit = {
    // check if a buffer is waiting
    if (callerBuffered(correlationId) != null) {
      // Iterate over the hashTable
      val test = callerBuffered(correlationId)
      for (array <- callerBuffered(correlationId)) {
        // Iterate over each (x, y) pair in the array
        val i = array
        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + i)
        val gradient = gradientsSync(correlationId)
        //   println("----------------------------------------------------------------------------------")
        //   println("buffered " + ucIndex + " : "  + correlationId + " " + layer + " " + i + " " + fromUCIndex)
        //   println("----------------------------------------------------------------------------------")
        actorHiddenLayer ! ComputeActivation.fromGradientsNeighbor(correlationId, gradient, layer, i, fromUCIndex)
      }
      callerBuffered -= (correlationId)
    }
    val x = Network.getHiddenLayersDim(layer, "hidden")
    for (i <- 0 until x if i != internalSubLayer) {
      val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + i)
      actorHiddenLayer ! ComputeActivation.getGradientsNeighbor(correlationId, gradients, layer, internalSubLayer, fromUCIndex)
    }
  }

  override def getNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, fromUCIndex: Int) : Unit = {
    val c = gradientsSync.contains(correlationId)
    // gradients not processed yet
    if (!c) {
      if (callerBuffered(correlationId) ==null)
        callerBuffered(correlationId) = Array(internalSubLayerCaller)
      else {
        callerBuffered(correlationId)  :+= internalSubLayerCaller
      }
    }
    else {
      val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + internalSubLayerCaller)
      actorHiddenLayer ! ComputeActivation.fromGradientsNeighbor(correlationId, gradientsSync(correlationId), layer, internalSubLayerCaller, fromUCIndex)
    }
  }

  override def FeedForwardTest(correlationId: String, shardedWeighted: Array[Float], internalSubLayer: Int, fromUCIndex:Int, layer: Int, shards: Int): Array[Float] = {
    counterFeedForward += 1
    if (layer == 2) {
      val test = 1
    }

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)

      var activation_tmp: Array[Float] = Array[Float]()
      var weighted_tmp: Array[Float] = Array[Float]()

      activation_tmp = Array.ofDim(LayerManager.GetDenseActivationLayerStep(layer))
      weighted_tmp = Array.fill(LayerManager.GetDenseActivationLayerStep(layer))(0)

      activation += (correlationId -> activation_tmp)
      weighted += (correlationId -> weighted_tmp)
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
    }
    val activationLength = bias.length
    if (layer == 3) {
      val test = 1
    }
    if (shardReceived(correlationId) < shards) {
      shardReceived(correlationId) += 1
      if (shardedWeighted.size == activationLength) {
        if (!weighted.contains(correlationId))
          weighted += (correlationId -> shardedWeighted)
        else
          weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), shardedWeighted)
      }
      else {
        if (shardReceived(correlationId) <= shards) {
          val biasLength = bias.length
          if (fromUCIndex == 0) {
            val act = shardedWeighted.padTo(biasLength, 0.0f)
            weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), act)
          }
          else if ((fromUCIndex + 1) < shards) {
            val index = getIndex(shards, biasLength, fromUCIndex)
            val test = Array.fill(biasLength)(0.0f)
            Array.copy(shardedWeighted, 0, test, index, shardedWeighted.length)
            weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), test)
          }
          else if ((fromUCIndex + 1) == shards) {
            val act2 = Array.fill(biasLength - shardedWeighted.length)(0.0f) ++ shardedWeighted
            weighted(correlationId) = MatrixHelper.matrixSum(weighted(correlationId), act2)
          }

        }
      }
    }


    //all received. Lets compute the activation function
    if (shards == shardReceived(correlationId) && inProgress(correlationId)) {
      val z = MatrixHelper.matrixSum(weighted(correlationId), bias)
      activation(correlationId) = ActivationManager.ComputeZ(Network.getActivationLayersType(layer), z)
      if (Network.LayerNorm) activation(correlationId) = MatrixHelper.layerNorm(activation(correlationId))

      if (Network.NaN) {
        activation(correlationId) = MatrixHelper.EliminateNaN(activation(correlationId))
      }

      inProgress(correlationId) = false
      shardReceived(correlationId) = 0

      //should we propagate to next hidden layer?
      val weightsTmp = activation(correlationId)
      //should we propagate to next hidden layer?
      val toUCs = Network.getHiddenLayersDim(layer+1, "weighted")

      for (i <- 0 until toUCs) {
        val actorweightedLayer = Network.LayersIntermediateRef("weightedLayer_" + (layer+1) + "_" + i)
        actorweightedLayer ! ComputeWeighted.FeedForwardTest(correlationId, activation(correlationId), i, layer+1)
      }
      shardReceived -= (correlationId)
      inProgress -= (correlationId)
      activation -= (correlationId)
      weighted -= (correlationId)
      minibatch -= (correlationId)
      weightsTmp
    }
    else
      null
  }
}
