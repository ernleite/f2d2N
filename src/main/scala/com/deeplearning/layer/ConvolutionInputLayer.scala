package com.deeplearning.layer

import breeze.linalg.{DenseVector, normalize}
import com.deeplearning.Network.heInitialization
import com.deeplearning.{ComputeActivation, MatrixHelper, LayerManager, Network, Normalisation}
import com.deeplearning.samples.{CifarData, MnistData, TrainingDataSet}

import java.time.{Duration, Instant}

class ConvolutionInputLayer extends InputLayer {
  var filters =  Array[Array[Float]]()
  var filtersCount = 0
  var size:Int = 0

  var kernels = ""
  var stride = 0
  var padding = ""
  var rows = 0
  var cols = 0
  var convolutionalFilter : ConvolutionalFilter = _
  var sharedNablas = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var sharedNabla = Array[Array[Float]]()
  var sharedWeighted = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var inputsConvolved = scala.collection.mutable.HashMap.empty[String, Array[Array[Float]]]
  var batchNorm = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private var dataSet: TrainingDataSet = if (Network.trainingSample == "Mnist") {
    dataSet = new MnistData()
    dataSet
  } else {
    dataSet = new CifarData()
    dataSet
  }
    def computeInputWeights(epoch: Int, correlationId: String, yLabel:Int, startIndex: Int, endIndex: Int, dataslice: Array[Float], index: Int, layer: Int, internalSubLayer:Int,params : scala.collection.mutable.HashMap[String,String]): Array[Array[Float]] = {
    if (lastEpoch != epoch) {
      counterTraining = 0
      counterBackPropagation = 0
      counterFeedForward = 0
      lastEpoch = epoch
    }
    epochCounter = epoch
    counterTraining += 1
    val nextLayer = layer + 1

    if (!wInitialized) {
      val arraySize = Network.InputLayerDim
      nabla_w = Array.ofDim(arraySize)
      dataSet.loadTrainDataset(Network.InputLoadMode)

      weights = Array.ofDim(Network.getHiddenLayersDim(nextLayer, "hidden"))
      nabla_w = Array.ofDim(Network.getHiddenLayersDim(nextLayer, "hidden"))
      this.size = this.dataSet.Size
      this.convolutionalFilter = new ConvolutionalFilter()
      this.size = Network.InputLayer/Network.channels
      this.convolutionalFilter.loadFilter(this.size,Network.Filters(layer),layer,dataSet.Channel,true)
      //should be a rounded value
      this.filters = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
    }

    if (!wInitialized) {
      wInitialized = true
      //Temporary
      // read shard data from data lake
      dataSet.loadTrainDataset(Network.InputLoadMode)

      //initialize the filters
      for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
       // val arr = heInitialization(this.convolutionalFilter.rows * this.convolutionalFilter.cols, 1, this.convolutionalFilter.stride)
        val arr2 = Network.generateRandomFloat(this.convolutionalFilter.rows * this.convolutionalFilter.cols)
        this.filters(i) = arr2
      }
    }

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      //for (i <- index until (index+1)) {
      val input = dataSet.getTrainingInput(index)
      val x = input.slice(2, dataSet.Size + 2) //normalisation
      val v = new DenseVector(x)
      val norm = normalize(v)
      this.X += (correlationId -> norm.toArray)
      var sharedNabla_Tmp : Array[Array[Float]] = Array(Array[Float]())
      var sharedNablas_Tmp : Array[Array[Float]] = Array(Array[Float]())
      var inputsConvolved : Array[Array[Float]] = Array(Array[Float]())

      sharedNablas_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNabla_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      inputsConvolved = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      this.inputsConvolved  += (correlationId -> inputsConvolved)
      this.sharedNabla = sharedNabla_Tmp
      this.sharedNablas += (correlationId -> sharedNablas_Tmp)
      wInitialized = true
    }

    val arrReshapes = this.X(correlationId).grouped(dataSet.Input2DRows*dataSet.Input2DCols).toArray
    var weighedfilters =  Array[Array[Float]]()
    weighedfilters = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
    var ind = 0
    //val startTime = Instant.now()

    for (i <- 0 until (this.convolutionalFilter.rangeCount)) {
      var shapes =  Array(Array(Array[Float]()))
      shapes = Array.ofDim(arrReshapes.length)
      val test =arrReshapes.indices.length
      for (j <- arrReshapes.indices) {
        val arrReshape = arrReshapes(j).grouped(dataSet.Input2DRows).toArray
        //val filterReshape = this.filters(ind).grouped(this.convolutionalFilter.rows).toArray
        //extract the channels
/*
        arrReshape(0)(0) = 1
        arrReshape(0)(1) = 2
        arrReshape(0)(2) = 3
        arrReshape(1)(0) = 4
        arrReshape(1)(1) = 5
        arrReshape(1)(2) = 6
        arrReshape(2)(0) = 7
        arrReshape(2)(1) = 8
        arrReshape(2)(2) = 9
        this.filters(0)(0) = 1
        this.filters(0)(1) = 2
        this.filters(0)(2) = 3
        this.filters(0)(3) = 4
        this.filters(0)(4) = 5
        this.filters(0)(5) = 6
        this.filters(0)(6) = 7
        this.filters(0)(7) = 8
        this.filters(0)(8) = 9

*/

        val (output1, output2) = ConvolutionalManager.convolution2D(arrReshape, this.filters(i))
        // sum the result
        if (j==0) {
          this.inputsConvolved(correlationId)(i) = output2.flatten
          weighedfilters(i) = output1.flatten
        }
        else {
          val test2 = output2.flatten
          val test1 = this.inputsConvolved(correlationId)(i)
          this.inputsConvolved(correlationId)(i) = MatrixHelper.matrixSum(this.inputsConvolved(correlationId)(i), output2.flatten)
          weighedfilters(i) =  MatrixHelper.matrixSum(weighedfilters(i),output1.flatten)
        }
      }
     // weighedfilters(i) = CostManager.batchNormalize(weighedfilters(i))
    }
      /*
      val endTime = Instant.now()
      val duration = Duration.between(startTime, endTime).toMillis
      if (counterTraining % Network.minibatchBuffer==0) {
        println("FilterFeedConvolution : " + duration)
      }
       */

      if (Network.ForwardBatchNormalization) {
        this.batchNorm += (correlationId -> weighedfilters.flatten)
        if (this.batchNorm.size == Network.MiniBatch) {
          val accArr = Normalisation.flattenHashMap(this.batchNorm)
          //group by filter
          val filters = accArr.grouped(this.convolutionalFilter.inputCols * this.convolutionalFilter.inputRows).grouped(Network.MiniBatch).toArray

          val concatenatedArrays: Array[Array[Float]] = (0 until Network.MiniBatch).map { subIndex =>
            filters.map { sequence =>
              sequence(subIndex)
            }.reduce(_ ++ _)
          }.toArray

          val keysIndexed = this.batchNorm.keys.toIndexedSeq

          val test = concatenatedArrays.zipWithIndex.map {
            case (sequence, index) =>
              val subArr = sequence
              val mav = Normalisation.getMeanAndVariance(subArr)
              val result = Normalisation.batchNormalize(subArr, mav._1, mav._3, 0.1f, 0.1f)
              val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + internalSubLayer)
              actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, keysIndexed(index), yLabel, Network.MiniBatchRange, result, internalSubLayer, 0, nextLayer, 1, null)
              val test = 1
          }

          this.batchNorm.clear()
        }
      }
      else {
        val weighted = weighedfilters.flatten
        //for (i <-0 until actorsDim) {
        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + internalSubLayer)
        actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, Network.MiniBatchRange, weighted, internalSubLayer,0,  nextLayer, 1, null)
      }
    weighedfilters
  }

  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Float, internalSubLayer: Int, fromInternalSubLayer: Int, params : scala.collection.mutable.HashMap[String,String]): Boolean = {
    //compute the derivative
    //context.log.info(s"Receiving backprogation request correlationId $correlationId input layer section ${internalSubLayer}")
    counterBackPropagation += 1
    minibatch(correlationId) += 1

    if (!backPropagateReceived.contains(correlationId)) {
      backPropagateReceived += (correlationId -> true)
    }

    val startTime = Instant.now()
    val deltaFilters = delta.grouped(this.convolutionalFilter.inputCols*this.convolutionalFilter.inputRows).toArray

    for (i <- 0 until (this.convolutionalFilter.rangeFilterCount)) {
      var shapes = Array(Array(Array[Float]()))
      shapes = Array.ofDim(deltaFilters.length)
      val inp1 = this.inputsConvolved(correlationId)(i).grouped(this.convolutionalFilter.rows*this.convolutionalFilter.cols).toArray

      // Create a new array to store the extracted first elements
      var newKernel = Array(Array[Float]())
      var newKernel2 = Array[Float]()
      newKernel = Array.ofDim(this.convolutionalFilter.rows*this.convolutionalFilter.cols)
      newKernel2 = Array.ofDim(this.convolutionalFilter.rows*this.convolutionalFilter.cols)

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

      this.sharedNablas(correlationId)(i) = newKernel2
    }

    val endTime = Instant.now()
    val duration = Duration.between(startTime, endTime).toMillis
    if ((Network.MiniBatch) == minibatch.values.sum) {

      sharedNabla = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      val grouped = this.sharedNablas.values.toArray.flatten.grouped(this.convolutionalFilter.rangeFilterCount).toArray
      this.sharedNabla = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      for (i <- 0 until this.convolutionalFilter.rangeFilterCount) {
        for (j <- 0 until Network.MiniBatch) {
          if (this.sharedNabla(i) == null)
            this.sharedNabla(i) = grouped(j)(i)
          else
            this.sharedNabla(i) = MatrixHelper.matrixSum(this.sharedNabla(i), grouped(j)(i))
        }

       // val tmp1 = CostManager.matMulScalar(( learningRate ), this.filters(i))
         val tmp2 = MatrixHelper.matMulScalar(learningRate , this.sharedNabla(i))
        //val tmp3 = CostManager.matMulScalar(learningRate / Network.MiniBatch, this.sharedNabla(i))
        this.filters(i) = MatrixHelper.minus(this.filters(i), tmp2)
        //this.filters(i) = CostManager.minus (tmp4, tmp3)
      }

      this.inputsConvolved.clear()
      backPropagateReceived.clear()
      minibatch.clear()
      weighted.clear()
      nablas_w.clear()
      sharedWeighted.clear()
      sharedNablas.clear()
      this.X.clear()
      true
    }
    else
      false
  }
  def FeedForwardTest(correlationId: String, startIndex: Int, endIndex: Int, index: Int, data: Array[Float], internalSubLayer: Int, layer: Int): Array[Array[Float]] = {
    if (!wTest) {
      wTest = true
      //Temporary
      // read shard data from data lake
      dataSet.loadTestDataset(Network.InputLoadMode)
    }
    val nextLayer = layer +1
    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      //for (i <- index until (index+1)) {
      val input = dataSet.getTestInput(index)
      val x = input.slice(2, dataSet.Size + 2) //normalisation
      val v = new DenseVector(x)
      val norm = normalize(v)
      this.X += (correlationId -> norm.toArray)
      //X = shuffleArray(X)
      val hiddenLayerStep = LayerManager.GetHiddenLayerStep(layer, "hidden")

      var weighted_tmp: Array[Array[Float]] = Array(Array[Float]())
      var nablasw_tmp: Array[Array[Array[Float]]] = Array(Array(Array[Float]()))
      var sharedNabla_Tmp: Array[Array[Array[Float]]] = Array(Array(Array[Float]()))
      var sharedNablas_Tmp: Array[Array[Array[Float]]] = Array(Array(Array[Float]()))

      sharedNablas_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
      sharedNabla_Tmp = Array.ofDim(this.convolutionalFilter.rangeFilterCount)

      weighted_tmp = Array.ofDim(Network.getHiddenLayersDim(nextLayer, "hidden"))
      nablasw_tmp = Array.ofDim(Network.getHiddenLayersDim(nextLayer, "hidden"))
      for (i <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
        weighted_tmp(i) = Array.fill[Float](hiddenLayerStep)(0)
        nablasw_tmp(i) = Array.ofDim(hiddenLayerStep)
      }

      //weighted += (correlationId -> weighted_tmp)
      wTest = true
    }

    val arrReshapes = this.X(correlationId).grouped(dataSet.Input2DRows * dataSet.Input2DCols).toArray
    var weighedfilters = Array[Array[Float]]()
    weighedfilters = Array.ofDim(this.convolutionalFilter.rangeFilterCount)
    var ind = 0
    for (i <- 0 until (this.convolutionalFilter.rangeCount)) {
      var shapes = Array(Array(Array[Float]()))
      shapes = Array.ofDim(arrReshapes.length)
      for (j <- arrReshapes.indices) {
        val arrReshape = arrReshapes(j).grouped(dataSet.Input2DRows).toArray
        //val filterReshape = this.filters(ind).grouped(this.convolutionalFilter.rows).toArray
        //extract the channels
        val (output1, output2) = ConvolutionalManager.convolution2D(arrReshape, this.filters(i))
        // sum the result
        if (j == 0) {
          weighedfilters(i) = output1.flatten
        }
        else {
          weighedfilters(i) = MatrixHelper.matrixSum(weighedfilters(i), output1.flatten)
        }
      }
    }

    val weightedTmp = weighedfilters.flatten
    val actorsDim = Network.HiddenLayersDim(nextLayer)

    //for (i <-0 until actorsDim) {
    val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + internalSubLayer)
    actorHiddenLayer ! ComputeActivation.FeedForwardTest( correlationId, weightedTmp, internalSubLayer, nextLayer,0, 1)
    minibatch -= (correlationId)
    X -= (correlationId)
    weighedfilters
  }
}
