package com.deeplearning.layer

import ai.djl.Device
import ai.djl.ndarray.{NDArray, NDManager}
import breeze.linalg.{DenseVector, normalize}
import com.deeplearning
import com.deeplearning.MatrixHelper.{dotProduct, dotProduct3, dotProduct4, dotProduct6}
import com.deeplearning.Network.heInitialization
import com.deeplearning.{ComputeActivation, ComputeWeighted, MatrixHelper, LayerManager, Network, Normalisation}


class ConvolutionWeightedLayer extends WeightedLayer {
  var sharedNablas = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var sharedNabla = Array[Array[Float]]()
  var sharedWeighted = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]

  var convolutionalFilter : ConvolutionalFilter = _
  var inputsConvolved = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var batchNorm = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var filters =  Array[Array[Float]]()
  var splitCount: Int = 0
  var size: Int = 0

  def Weights(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, activations: Array[Float], shards:Int, internalSubLayer: Int, layer: Int, params : scala.collection.mutable.HashMap[String,String]) : Array[Array[Float]] = {
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

    if (!LayerManager.IsLast(nextLayer)) {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(nextLayer, "weighted")
      arraySize = Network.getHiddenLayersDim(nextLayer, "weighted")
    }
    else {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
    }

    if (!this.wInitialized) {
      this.convolutionalFilter = new ConvolutionalFilter()
      val channels = ConvolutionalFilterHelper.getChannelSize(layer - 1)
      this.splitCount = ConvolutionalFilterHelper.getSplitSize(layer - 1)
      this.size = activations.grouped(channels).size

      this.convolutionalFilter.loadFilter(this.size, Network.Filters(layer-1), layer , channels,true)
      this.filters = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      if (this.convolutionalFilter.convolution) {
        for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
          //val arr = heInitialization(this.convolutionalFilter.rows * this.convolutionalFilter.cols, 1, this.convolutionalFilter.stride)

          val arr2 = Network.generateRandomFloat(this.convolutionalFilter.rows * this.convolutionalFilter.cols)
          this.filters(i) = arr2
        }
      }

      this.weights = Array.ofDim(arraySize)
      this.nabla_w = Array.ofDim(arraySize)
      var sharedNabla_Tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedNablas_Tmp: Array[Array[Float]] = Array(Array[Float]())
      var inputsConvolved: Array[Array[Float]] = Array(Array[Float]())

      sharedNablas_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNabla_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      inputsConvolved = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      this.inputsConvolved += (correlationId -> inputsConvolved)
      this.sharedNabla = sharedNabla_Tmp
      this.sharedNablas += (correlationId -> sharedNablas_Tmp)

      this.wInitialized = true
    }

    //this.activation += (correlationId -> activations)
    if (!this.minibatch.contains(correlationId)) {
      var inputsConvolved: Array[Array[Float]] = Array(Array[Float]())
      inputsConvolved = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      this.inputsConvolved += (correlationId -> inputsConvolved)
      this.minibatch += (correlationId -> 0)
      this.messagePropagateReceived += (correlationId -> 0)
      this.fromInternalReceived += (correlationId -> 0)
      this.activation += (correlationId -> activations)

      var sharedNabla_Tmp: Array[Array[Float]] = Array(Array[Float]())
      var sharedNablas_Tmp: Array[Array[Float]] = Array(Array[Float]())
      sharedNablas_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNabla_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      inputsConvolved = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      this.sharedNabla = sharedNabla_Tmp
      this.sharedNablas += (correlationId -> sharedNablas_Tmp)
    }

    if (convolutionalFilter.convolution) {
      val arrReshapes = activations.grouped(this.size).toArray
      var weighedfilters = Array[Array[Float]]()
      weighedfilters = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      var ind = 0
      //val startTime = Instant.now()

      for (i <- 0 until (this.convolutionalFilter.rangeCount)) {
        var shapes = Array(Array(Array[Float]()))
        shapes = Array.ofDim(arrReshapes.length)

        for (j <- arrReshapes.indices) {
          val v = new DenseVector(arrReshapes(j))
          val norm = normalize(v).toArray

          val arrReshape = norm.grouped(ConvolutionalFilterHelper.getRowsSize(this.size)).toArray
          //val filterReshape = this.filters(ind).grouped(this.convolutionalFilter.rows).toArray
          //extract the channels
          val (output1, output2) = ConvolutionalManager.convolution2D(arrReshape, this.filters(i))
          // sum the result
          if (j == 0) {
            this.inputsConvolved(correlationId)(i) = output2.flatten
            weighedfilters(i) = output1.flatten
          }
          else {
            this.inputsConvolved(correlationId)(i) = MatrixHelper.matrixSum(this.inputsConvolved(correlationId)(i), output2.flatten)
            weighedfilters(i) = MatrixHelper.matrixSum(weighedfilters(i), output1.flatten)
          }
        }
       // weighedfilters(i) = CostManager.batchNormalize(weighedfilters(i))
      }

      if (Network.ForwardBatchNormalization) {
        this.batchNorm += (correlationId -> weighedfilters.flatten)
        if (this.batchNorm.size == Network.MiniBatch) {
          val accArr = Normalisation.flattenHashMap(this.batchNorm)
          //group by filter
          val filters = accArr.grouped(this.convolutionalFilter.inputCols*this.convolutionalFilter.inputRows).grouped(Network.MiniBatch).toArray

          val concatenatedArrays: Array[Array[Float]] = (0 until Network.MiniBatch).map { subIndex =>
            filters.map { sequence =>
              sequence(subIndex)
            }.reduce(_ ++ _)
          }.toArray

          val keysIndexed = this.batchNorm.keys.toIndexedSeq

          concatenatedArrays.zipWithIndex.map {
            case (sequence, index) =>
              val subArr = sequence
              val mav = Normalisation.getMeanAndVariance(subArr)
              val result = Normalisation.batchNormalize(subArr, mav._1,mav._3, 0.01f, 0f)
              val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + internalSubLayer)
              actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, keysIndexed(index), yLabel, Network.MiniBatchRange, result, internalSubLayer, internalSubLayer, layer, 1,null)
          }
          this.batchNorm.clear()
        }
      }
      else {
        val weighted = weighedfilters.flatten
        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + internalSubLayer)
        actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, Network.MiniBatchRange, weighted, internalSubLayer, internalSubLayer,  layer, 1,null)
      }

      weighedfilters
    }
    else {
    // create a context and command queue for the GPU
    // create a tensor from the array
    // send to actors for Z compute
      for (i <- 0 until arraySize) {
        this.weighted(correlationId) = dotProduct(this.weights, activations)
      }
      this.weighted(correlationId).grouped(1).toArray
    }
  }

  def BackPropagate(correlationId: String, delta: Array[Float],learningRate: Float, regularisation: Float, nInputs: Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int, params : scala.collection.mutable.HashMap[String,String], applyGrads: Boolean) : Boolean = {
    this.counterBackPropagation += 1

    var hiddenLayerStep = 0
    var arraySize = 0

    this.minibatch(correlationId) += 1
    this.messagePropagateReceived(correlationId) += 1
    val fromLayer = layer + 1
    var fromArraySize = 0
    if (fromLayer == (Network.HiddenLayersDim.length+1)) {
      fromArraySize = Network.OutputLayerDim
    }
    else {
      fromArraySize = Network.getHiddenLayersDim(fromLayer, "weighted")
    }

    if (!backPropagateReceived.contains(correlationId)) {
      backPropagateReceived += (correlationId -> true)
      sharedNablas(correlationId) = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
    }

    //compute new delta
    val nextlayer = layer + 1
    var callerSize = 0
    if (LayerManager.IsLast(layer)) {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
    }
    else {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(nextlayer, "weighted")
      arraySize = Network.getHiddenLayersDim(nextlayer, "weighted")
    }

    if (this.convolutionalFilter.convolution) {
      //val deltaFilters = delta.grouped(this.convolutionalFilter.rows * this.convolutionalFilter.cols).toArray
      val deltaFilters = delta.grouped(this.convolutionalFilter.inputCols*this.convolutionalFilter.inputRows).toArray
      val deltaWeights:Array[Array[Float]] = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      val acti:Array[Array[Float]] = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      for (i <- 0 until (this.convolutionalFilter.rangeFilterCount)) {
        var shapes = Array(Array(Array[Float]()))
        shapes = Array.ofDim(deltaFilters.length)
        val inp1 = this.inputsConvolved(correlationId)(i).grouped(this.convolutionalFilter.rows * this.convolutionalFilter.cols).toArray

        // Create a new array to store the extracted first elements
        var newKernel = Array(Array[Float]())
        var newKernel2 = Array[Float]()
        newKernel = Array.ofDim(this.convolutionalFilter.rows * this.convolutionalFilter.cols)
        newKernel2 = Array.ofDim(this.convolutionalFilter.rows * this.convolutionalFilter.cols)

        // Iterate through the rows and extract the first element
        for (j <- 0 until newKernel.length) {
          newKernel(j) = Array.ofDim(inp1.length)
          for (k <- 0 until inp1.length) {
            val row: Array[Float] = inp1(k)
            val firstElement: Float = row(j)
            newKernel(j)(k) = firstElement
          }
          newKernel2(j) = MatrixHelper.dotProduct(newKernel(j), deltaFilters(i)).sum
        }
        //nablas_w(correlationId)(fromInternalSubLayer) = dotProduct3(delta.length, delta, activation(correlationId))
        //val newdelta = dotProduct4(LayerManager.GetDenseLayerStep(layer), weights(fromInternalSubLayer), delta)
        this.sharedNablas(correlationId)(i) = newKernel2
      }

      val act = activation(correlationId)
      val dl = delta
     // val newdelta = dotProduct(activation(correlationId), delta)

      val previousLayer = layer -1
      val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + previousLayer + "_" + internalSubLayer)
      fromInternalReceived(correlationId) += 1
      hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, this.sharedNablas(correlationId).flatten, learningRate, regularisation, nInputs, 0, previousLayer, internalSubLayer, internalSubLayer, params)

      if ((Network.MiniBatch) == minibatch.values.sum) {

        sharedNabla = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
        /*
        for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
          sharedNabla(i) = Array.fill[Float](this.convolutionalFilter.rows * this.convolutionalFilter.cols)(0.0f)
          for ((k, w) <- this.sharedNablas) {
            sharedNabla(i) = (DenseVector(sharedNabla(i)) + DenseVector(sharedNablas(k)(i))).toArray
          }
        }
        */
        val grouped = this.sharedNablas.values.toArray.flatten.grouped(this.convolutionalFilter.rangeFilterCount).toArray
        this.sharedNabla = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
        for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
          for (j <- 0 until Network.MiniBatch) {
            if (this.sharedNabla(i) == null)
              this.sharedNabla(i) = grouped(j)(i)
            else
              this.sharedNabla(i) = MatrixHelper.matrixSum(this.sharedNabla(i), grouped(j)(i))
          }

          val tmp1 = MatrixHelper.matMulScalar((1 - learningRate * (regularisation / nInputs)), this.filters(i))
          val tmp2 = MatrixHelper.matMulScalar(learningRate / Network.MiniBatch, this.sharedNabla(i))
          //val tmp3 = CostManager.matMulScalar(learningRate / Network.MiniBatch, this.sharedNabla(i))
          this.filters(i) = MatrixHelper.minus(tmp1, tmp2)
        }


        this.inputsConvolved.clear()
        backPropagateReceived.clear()
        minibatch.clear()
        weighted.clear()
        nablas_w.clear()
        sharedWeighted.clear()
        sharedNablas.clear()
        true
      }
      else
        false

    }
    else {
      val newdelta = dotProduct(weights, delta)

      callerSize = arraySize * Network.getHiddenLayersDim(layer, "weighted")
      // only send when all the call from the layer +1 are received
      if (fromArraySize == messagePropagateReceived(correlationId)) {
        val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + internalSubLayer)
        fromInternalReceived(correlationId) += 1
        hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, newdelta, learningRate, regularisation, nInputs, 0, layer, internalSubLayer,internalSubLayer, params)
      }

      /*
      // check if we reach the last mini-bacth
      if (Network.MiniBatch * fromArraySize == minibatch.values.sum) {

        //[ (1-eta*(lmbda/n))  *  w  -  (eta/len(mini_batch))  *nw
        //nabla_w = Array.ofDim(arraySize)
        for (i <- 0 until arraySize) {
          nabla_w(i) = Array.tabulate(hiddenLayerStep, this.activation(correlationId).length)((_, _) => 0)
          for ((k, w) <- nablas_w) {
            if (layer == 0) {
              val test = 1
            }
            nabla_w(i) = (DenseVector(nabla_w(i)) + DenseVector(nablas_w(k)(i))).toArray
          }
        }

        for (i <- 0 until arraySize) {
          for (j <- 0 until hiddenLayerStep) {
            val tmp = this.weights(i)(j)
            val tmp1 = (1 - learningRate * (regularisation / nInputs)) * DenseVector(tmp)
            val tmp2 = (learningRate / Network.MiniBatch) * DenseVector(nabla_w(i)(j))
            weights(i)(j) = (tmp1 - tmp2).toArray
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

       */
      false
    }

  }

  override def FeedForwardTest(correlationId: String, activations: Array[Float], internalSubLayer: Int, layer: Int): Array[Array[Float]] =  {
    val nextLayer = layer + 1
    counterFeedForward += 1

    var hiddenLayerStep = 0
    var arraySize = 0

    if (!LayerManager.IsLast(nextLayer)) {
      hiddenLayerStep = LayerManager.GetHiddenLayerStep(nextLayer, "weighted")
      arraySize = Network.getHiddenLayersDim(nextLayer, "weighted")
    }
    else {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
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
     // this.weighted += (correlationId -> weighted_tmp)
    }

    /*
    // send to actors for Z compute
    if (!LayerManager.IsLast(nextLayer)) {
      for (i <- 0 until arraySize) {
        this.weighted(correlationId)(i) = dotProduct6(LayerManager.GetHiddenLayerStep(layer), this.weights(i), activations)
      }
    }
    else  {
      for (i <- 0 until Network.OutputLayerDim) {
        this.weighted(correlationId)(i) = dotProduct6(LayerManager.GetHiddenLayerStep(layer), this.weights(i), activations)
      }
    }
    */
    val tmp = this.weighted(correlationId)
    //val weighted = tmp.flatten
    //val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + internalSubLayer)
    //actorHiddenLayer ! ComputeActivation.FeedForwardTest(correlationId, weighted, internalSubLayer, layer, 1)

    this.activation -= correlationId
    this.weighted -= correlationId
    this.minibatch -= correlationId

    tmp.grouped(1).toArray
  }

  override def getNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayer:Int,localIndex:Int): Unit = {
    weighted(correlationId)
  }

  def SynchronizeWeights(correlationId:String, gradients:Array[Float], layer:Int, internalSubLayer:Int, fromInternalSubLayer: Int): Unit = {
  }

  override def sendGradients(correlationId: String, gradients: Array[Float], layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int): Unit = {

  }

  override def applyGradients(correlationId: String, gradients: Array[Float], layer: Int,  internalSubLayer:Int, fromInternalSubLayer: Int): Unit = {

  }
}
