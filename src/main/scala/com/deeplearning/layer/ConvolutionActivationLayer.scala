package com.deeplearning.layer

import com.deeplearning.CostManager.dotProduct
import com.deeplearning.{ActivationManager, ComputeInputs, ComputeWeighted, CostManager, LayerManager, Network}
class ConvolutionActivationLayer extends ActivationLayer {

  var convolutionalFilter: ConvolutionalFilter = _
  var sharedNablas = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var sharedWeighted = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var sharedFeatureMap = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var sharedZ = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var sharedActivation = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var sharedbias = Array[Float]()
  var sharedNabla = Array[Array[Float]]()
  var poolIndices = scala.collection.mutable.HashMap.empty[String, Array[Array[(Int, Int)]]]
  var splitCount: Int = 0
  var size: Int = 0

  override def ComputeZ(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, shardedWeighted: Array[Float], internalSubLayer: Int,fromInternalSubLayer:Int, layer: Int, shards: Int, params: scala.collection.mutable.HashMap[String, String]): Array[Float] = {
    if (lastEpoch != epoch) {
      counterTraining = 0
      counterBackPropagation = 0
      counterFeedForward = 0
      lastEpoch = epoch
    }

    // bias initialized only one time during the training cycle
    counterTraining += 1
    if (!bInitialized) {
      //this.bias = generateRandomBiasFloat(shardedWeighted.length)
      this.convolutionalFilter = new ConvolutionalFilter()
      var channels = ConvolutionalFilterHelper.getChannelSize(layer - 1)
      this.splitCount = ConvolutionalFilterHelper.getSplitSize(layer - 1)
      this.size = shardedWeighted.grouped(this.splitCount).size
      this.convolutionalFilter.loadFilter(this.size, Network.Filters(layer - 1), layer - 1, channels, false)
      this.convolutionalFilter.splitCount = this.convolutionalFilter.filtersCount / shards
      this.convolutionalFilter.filteredCount = this.convolutionalFilter.filtersCount * this.convolutionalFilter.rangeFilterCount
      //should be a rounded value
      this.sharedbias = Array.fill[Float](this.convolutionalFilter.rangeFilterCount)(0.0f)

      bInitialized = true
    }


    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      messagePropagateReceived += (correlationId -> 0)

      var sharedFeatureMap_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedWeighted_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedNablas_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedNabla_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedActivation_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedZ_tmp: Array[Array[Float]] = Array(Array[Float]())
      var poolIndices_tmp: Array[Array[(Int, Int)]] = Array(Array[(Int, Int)]())

      sharedFeatureMap_tmp = Array.ofDim(shards)
      sharedWeighted_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedActivation_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedZ_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNablas_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNabla_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      poolIndices_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      sharedFeatureMap += (correlationId -> sharedFeatureMap_tmp)
      poolIndices += (correlationId -> poolIndices_tmp)
      sharedWeighted += (correlationId -> sharedWeighted_tmp)
      sharedActivation += (correlationId -> sharedActivation_tmp)
      sharedNablas += (correlationId -> sharedNablas_tmp)
      sharedZ += (correlationId -> sharedZ_tmp)
      sharedNabla = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
    }
    if (layer == 2) {
      val test = 1
    }
    sharedFeatureMap(correlationId)(shardReceived(correlationId)) = shardedWeighted
    shardReceived(correlationId) += 1

    //all received. Lets compute the activation function
    if (shards == shardReceived(correlationId) && inProgress(correlationId)) {
      //val normalizeWeights = Normalization.forward(weighted(correlationId), weighted.size)
      this.weighted += (correlationId -> shardedWeighted)
      val range = (this.convolutionalFilter.rangeFilterCount)
      var ind = 0
      var pool = Array(Array[Float]())
      pool = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      val test = sharedFeatureMap(correlationId).length
      //number of split of the layer
      for (i <- 0 until sharedFeatureMap(correlationId).length) {
        var featureMapConcat: Array[Float] = Array[Float]()
        //number of split from the previous layer
        //concat to have a concatened feature map
        //number of split from the matrix received
        val currents = sharedFeatureMap(correlationId)(i).grouped(this.convolutionalFilter.inputRows * this.convolutionalFilter.inputCols).toArray
        for (y <- 0 until currents.length) {
          var shapes = Array(Array(Array[Float]()))
          shapes = Array.ofDim(shards)
          featureMapConcat = currents(i)
          var activationfilters: Array[Float] = Array.fill[Float](this.convolutionalFilter.inputRows * this.convolutionalFilter.inputCols)(0.0f)
          this.sharedZ(correlationId)(y) = featureMapConcat

          val currenty = currents(y)
          val currntbias = this.sharedbias(y)

          activationfilters = CostManager.sum(currents(y), this.sharedbias(y))
          if (Network.dropout > 0) {
            activationfilters = Network.dropout(activationfilters)
          }
          if (Network.NaN) {
            activationfilters = CostManager.EliminateNaN(activationfilters)
          }

          this.sharedActivation(correlationId)(y) = ActivationManager.ComputeZ(Network.getActivationLayersType(layer), activationfilters)

          if (this.convolutionalFilter.pooling) {
            val grouped = activationfilters.grouped(this.convolutionalFilter.inputRows).toArray
            //val matrix: DenseMatrix[Float] = DenseMatrix.create[Float](grouped.length, grouped(0).length, grouped.flatten).t
            val (pooling, poolIndices) = this.convolutionalFilter.pooling(convolutionalFilter, grouped)
            pool(y) = pooling.flatten
            this.poolIndices(correlationId)(y) = poolIndices
          }
          else {
            pool(y) = this.sharedActivation(correlationId)(y)
          }
        }
      }

      //need to implement vertical parallelism
      inProgress(correlationId) = false
      shardReceived(correlationId) = 0
      this.sharedFeatureMap.clear()
      //should we propagate to next hidden layer?
      val nextLayer = layer + 1
      if (nextLayer == 2) {
        val test = 1
      }
      val actorweightedLayer = Network.LayersIntermediateRef("weightedLayer_" + nextLayer + "_" + internalSubLayer)
      //todo send parameters (min, max, etc)
      actorweightedLayer ! ComputeWeighted.Weights(epoch, correlationId, yLabel, trainingCount, pool.flatten, 1, internalSubLayer, nextLayer, null)

      pool.flatten

    }
    else
      null
  }

  override def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, shards:Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer:Int, params: scala.collection.mutable.HashMap[String, String]): Boolean = {
    counterBackPropagation += 1
    //compute the derivative
    //context.log.info(s"Receiving backprogation request correlationId $correlationId HiddenLayer_${layer}_${internalSubLayer}")
    minibatch(correlationId) += 1
    messagePropagateReceived(correlationId) += 1

    if (!backPropagateReceived.contains(correlationId)) {
      backPropagateReceived += (correlationId -> true)
    }
    if (layer == 2) {
      val test = 1
    }
    if (layer == 1) {
      val test = 1
    }

    var deltaGrouped: Array[Array[Float]] = Array[Array[Float]]()
    if (this.convolutionalFilter.pooling) {
      deltaGrouped = delta.grouped(this.convolutionalFilter.poolRows * this.convolutionalFilter.poolCols).toArray
    }
    else {
      deltaGrouped = delta.grouped(this.convolutionalFilter.inputRows * this.convolutionalFilter.inputCols).toArray
    }

    //test transpose pooling > feature map
    val empty = Array.fill[Float](this.weighted(correlationId).length)(0f)
    val grouped = empty.grouped(this.convolutionalFilter.inputRows * this.convolutionalFilter.inputCols).toArray

    var deltaFinal = Array[Array[Float]]()
    deltaFinal = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

    val weightedTmp = this.weighted(correlationId)
    val weightedGrouped = weightedTmp.grouped(this.convolutionalFilter.inputRows * this.convolutionalFilter.inputCols).toArray

    for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
      val prime = ActivationManager.ComputePrime(Network.getActivationLayersType(layer), this.sharedZ(correlationId)(i))
      val tmpDelta = deltaGrouped(i)

      if (convolutionalFilter.pooling) {
        var emptyGrouped = grouped(i).grouped(this.convolutionalFilter.inputRows).toArray
        var ind = 0
        var ind2 = 0
        convolutionalFilter.poolingStrategy match {
          case "max" =>
            for (j <- this.poolIndices(correlationId)(i).indices) {
              val tuple = this.poolIndices(correlationId)(i)(j)
              emptyGrouped(tuple._1)(tuple._2) = tmpDelta(ind)
              ind = ind + 1
            }

          case "avg" =>
            for (j <- this.poolIndices(correlationId)(i).indices) {
              val tuple = this.poolIndices(correlationId)(i)(j)
              emptyGrouped(tuple._1)(tuple._2) = tmpDelta(ind2)
              if (ind % (convolutionalFilter.poolDimRow * convolutionalFilter.poolDimCol) == 0)
                ind2 = ind2 + 1
              ind += 1
            }
        }
        deltaFinal(i) = emptyGrouped.flatten

        var deltaprime = dotProduct(prime, deltaFinal(i))
        if (Network.NaN) {
          deltaprime = CostManager.EliminateNaN(deltaprime)
        }
        this.sharedNablas(correlationId)(i) = deltaprime
      }
      else {
        val deltaprime = dotProduct(prime, weightedGrouped(i))
        this.sharedNablas(correlationId)(i) = deltaprime
      }
      //upsampling
    }

    val previousLayer = layer - 1
    var hiddenLayerStep = 0
    var fromArraySize = 0

    val nextlayer = layer + 1
    var callerSize = 0
    var arraySize = 0
    if (LayerManager.IsLast(layer)) {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      fromArraySize = Network.OutputLayerDim
    }
    else {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(nextlayer, "hidden")
      fromArraySize = Network.getHiddenLayersDim(nextlayer, "hidden")
    }

    if (LayerManager.IsFirstConvolution(layer)) {
      hiddenLayerStep = LayerManager.GetInputLayerStep()
      arraySize = Network.InputLayerDim
      //context.log.info(s"$correlationId Inputlayer reached $layer $internalSubLayer " + fromArraySize+ " " + messagePropagateReceived.map(r => r._2).sum)
      //  for (i <- 0 until arraySize) {
      val inputLayerRef = Network.LayersInputRef("inputLayer_" + internalSubLayer)
      //we need to flatten the range of sharedActivation and to send it back to previous layer

      val sharedDeltaPrime = this.sharedNablas(correlationId).flatten
      inputLayerRef ! ComputeInputs.BackPropagate(correlationId, sharedDeltaPrime, learningRate, regularisation, nInputs.toFloat, internalSubLayer, internalSubLayer, params)
      //      }
    }
    else {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(previousLayer, "hidden")
      arraySize = Network.getHiddenLayersDim(previousLayer, "hidden")

      // for (i <- 0 until arraySize) {
      //context.log.info(s"$correlationId Send backprogragation intermediate layer $previousLayer " + i)
      val intermediateLayerRef = Network.LayersIntermediateRef("weightedLayer_" + layer + "_" + internalSubLayer)
      val sharedDeltaPrime = this.sharedNablas(correlationId).flatten
      intermediateLayerRef ! ComputeWeighted.BackPropagate(correlationId, sharedDeltaPrime, learningRate, regularisation, nInputs, layer, internalSubLayer, internalSubLayer, params)
      //}
    }

    // check if we reach the last mini-bacth
    if ((Network.MiniBatch) == minibatch.values.sum) {
      /*
      for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
        this.sharedNabla(i) = Array.fill[Float](this.convolutionalFilter.inputRows*this.convolutionalFilter.inputCols)(0.0f)
      }

      for ((k, w) <- backPropagateReceived) {
        for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
          this.sharedNabla(i) = (DenseVector(this.sharedNabla(i)) + DenseVector(this.sharedNablas(k)(i))).toArray
        }
      }

        self.biases = [b-(eta/len(mini_batch))*nb
                      for b, nb in zip(self.biases, nabla_b)]
       */
      val grouped = this.sharedNablas.values.toArray.flatten.grouped(this.convolutionalFilter.rangeFilterCount).toArray
      this.sharedNabla = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
        for (j <- 0 until Network.MiniBatch) {
          if (this.sharedNabla(i) == null)
            this.sharedNabla(i) = grouped(j)(i)
          else
            this.sharedNabla(i) = CostManager.sum2(this.sharedNabla(i), grouped(j)(i))
        }
        val tmp2 = CostManager.matMulScalar(learningRate, this.sharedNabla(i))
        this.sharedbias(i) = this.sharedbias(i) - tmp2.sum
      }
      /*
      val tmp2 = (learningRate / Network.MiniBatch) * DenseVector(nabla_b)
      val tmp = DenseVector(this.bias)
      bias = (tmp - tmp2).toArray
  */

      messagePropagateReceived.clear()
      poolIndices.clear()
      nablas_b.clear()
      activation.clear()
      backPropagateReceived.clear()
      weighted.clear()
      sharedWeighted.clear()
      activation.clear()
      inProgress.clear()
      shardReceived.clear()
      this.sharedNablas.clear()
      this.sharedActivation.clear()
      this.sharedZ.clear()
      minibatch.clear()
      Z.clear()
      true
    }
    else
      false
  }

  override def FeedForwardTest(correlationId: String, shardedWeighted: Array[Float], internalSubLayer: Int, fromInternalSubLayer:Int, layer: Int, shards: Int): Array[Float] = {
    counterFeedForward += 1

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      var sharedWeighted_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedNablas_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedNabla_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedActivation_tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedZ_tmp: Array[Array[Float]] = Array(Array[Float]())

      sharedWeighted_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedActivation_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedZ_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNablas_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNabla_tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      var sharedFeatureMap_tmp: Array[Array[Float]] = Array(Array[Float]())
      sharedFeatureMap_tmp = Array.ofDim(shards)
      sharedFeatureMap += (correlationId -> sharedFeatureMap_tmp)
      sharedWeighted += (correlationId -> sharedWeighted_tmp)
      sharedActivation += (correlationId -> sharedActivation_tmp)
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
    }
    sharedFeatureMap(correlationId)(shardReceived(correlationId)) = shardedWeighted
    shardReceived(correlationId) += 1

    //all received. Lets compute the activation function
    if (shards == shardReceived(correlationId) && inProgress(correlationId)) {
      //val normalizeWeights = Normalization.forward(weighted(correlationId), weighted.size)
      this.weighted += (correlationId -> shardedWeighted)
      val range = (this.convolutionalFilter.rangeFilterCount)
      var ind = 0
      var pool = Array(Array[Float]())
      pool = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      //number of split of the layer
      for (i <- 0 until sharedFeatureMap(correlationId).length) {
        var featureMapConcat: Array[Float] = Array[Float]()
        //number of split from the previous layer
        //concat to have a concatened feature map
        //number of split from the matrix received
        val currents = sharedFeatureMap(correlationId)(i).grouped(this.convolutionalFilter.inputRows * this.convolutionalFilter.inputCols).toArray
        for (y <- 0 until this.splitCount) {
          var shapes = Array(Array(Array[Float]()))
          shapes = Array.ofDim(shards)
          featureMapConcat = currents(i)
          var activationfilters: Array[Float] = Array.fill[Float](this.convolutionalFilter.inputRows * this.convolutionalFilter.inputCols)(0.0f)
          //          this.sharedZ(correlationId)(y) = featureMapConcat

          activationfilters = CostManager.sum(currents(y), this.sharedbias(y))
          if (Network.dropout > 0) {
            activationfilters = Network.dropout(activationfilters)
          }
          if (Network.NaN) {
            activationfilters = CostManager.EliminateNaN(activationfilters)
          }

          this.sharedActivation(correlationId)(y) = ActivationManager.ComputeZ(Network.getActivationLayersType(layer), activationfilters)

          if (this.convolutionalFilter.pooling) {
            val grouped = activationfilters.grouped(this.convolutionalFilter.inputRows).toArray
            //val matrix: DenseMatrix[Float] = DenseMatrix.create[Float](grouped.length, grouped(0).length, grouped.flatten).t
            val (pooling, poolIndices) = this.convolutionalFilter.pooling(convolutionalFilter, grouped)
            pool(y) = pooling.flatten
          }
          else {
            pool(y) = this.sharedActivation(correlationId)(y)
          }
        }
      }

      //need to implement vertical parallelism
      inProgress(correlationId) = false
      shardReceived(correlationId) = 0

      //should we propagate to next hidden layer?
      val weightsTmp = pool.flatten

      val nextLayer = layer + 1
      val actorweightedLayer = Network.LayersIntermediateRef("weightedLayer_" + nextLayer + "_" + internalSubLayer)
      actorweightedLayer ! ComputeWeighted.FeedForwardTest(correlationId, pool.flatten, internalSubLayer, nextLayer)

      this.sharedFeatureMap.clear()
      this.sharedActivation -= (correlationId)
      this.sharedWeighted -= (correlationId)
      shardReceived -= (correlationId)
      inProgress -= (correlationId)
      minibatch -= (correlationId)
      weightsTmp
    }
    else
      null
  }

  override def getNeighbor(correlationId: String, gradients: Array[Float], layer: Int, internalSubLayer: Int, localIndex: Int): Unit = {
    weighted(correlationId)
  }

  def SynchronizeWeights(correlationId: String, gradients: Array[Float], layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int): Unit = {
  }

  override def sendGradients(correlationId: String, gradients: Array[Float], layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int): Unit = {

  }

  override def applyGradients(correlationId: String, gradients: Array[Float], layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int): Unit = {

  }
}
