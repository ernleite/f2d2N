package com.deeplearning.layer

import breeze.linalg.DenseMatrix
import com.deeplearning.Network

object ConvolutionalFilterHelper {
  def getSplitSize(layer:Int): Int = {
    val filter = Network.Filters(layer)
    val filterParams = filter.split(";")
    val filterCount = filterParams(0).split(":")(1).toInt
    if (layer == 0)
      filterCount/Network.InputLayerDim
    else
      filterCount/Network.HiddenLayersDim(layer)
  }

  def getRowsSize(length: Int): Int = {
    if (length>0)
      math.sqrt(length).toInt
    else 0
  }

  def getChannelSize(layer:Int): Int = {
    if (layer == 0)
      Network.channels
    else{
      val filter = Network.Filters(layer-1)
      val filterParams = filter.split(";")
      val filterCount = filterParams(0).split(":")(1).toInt
      filterCount
    }
  }
}

class ConvolutionalFilter {
  var inputRows = 0
  var inputCols = 0
  var poolRows = 0
  var poolCols = 0
  var rows = 0
  var cols = 0
  var stride = 0
  var filtersCount = 0
  var padding = ""
  var rangeFilterCount = 0
  var rangeCount = 0
  var splitCount = 0
  var filteredCount = 0
  var layer = 0
  var channels = 0
  var convolution = false
  var pooling: Boolean = false
  var poolingStrategy: String = "max"
  var poolDimRow = 2
  var poolDimCol = 2

  def loadFilter(inputLength:Int, filter:String,layer:Int, channels:Int, applyConvolution:Boolean) : Unit = {
    //define the number of channels
    this.channels = Network.channels
    this.inputRows = math.sqrt(inputLength).toInt
    this.inputCols = this.inputRows
    this.layer = layer
    this.channels = channels
    if (!filter.isEmpty) {
      this.convolution = true
      val filterParams = filter.split(";")
      this.filtersCount = filterParams(0).split(":")(1).toInt
      this.rows = filterParams(1).split(":")(1).split(",")(0).toInt
      this.cols = filterParams(1).split(":")(1).split(",")(1).toInt
      this.stride = filterParams(2).split(":")(1).toInt
      this.rangeCount = this.filtersCount / Network.getLayersDim(layer)
      this.rangeFilterCount = this.rangeCount
      if (applyConvolution) {
        this.inputRows = this.inputRows - this.rows + 1
        this.inputCols = this.inputCols - this.cols + 1
      }
      val poolingPattern = filter.split("pooling:")
      if (poolingPattern.length > 1) {
        val pooling = poolingPattern(1).split(";")
        this.pooling = true
        val tuple = pooling(0).split(",")
        this.poolingStrategy = tuple(0)
        this.poolDimRow = tuple(1).toInt
        this.poolDimCol = tuple(2).toInt
        this.poolRows = this.inputRows/this.poolDimRow
        this.poolCols = this.inputCols/this.poolDimCol
      }
    }
  }

  /**
   * Performs the transposed convolution operation on the input feature map.
   *
   * @param input          The input feature map.
   * @param weights        The weights of the transposed convolution kernel.
   * @param outputChannels The number of output channels.
   * @param stride         The stride of the transposed convolution.
   * @param padding        The padding of the transposed convolution.
   * @return The output feature map after the transposed convolution.
   */
  def ConvTranspose2D(input: DenseMatrix[Float], weights: DenseMatrix[Float], outputChannels: Int, stride: (Int, Int), padding: (Int, Int)): DenseMatrix[Float] = {
    val (inputRows, inputCols) = (input.rows, input.cols)
    val (kernelRows, kernelCols) = (weights.rows, weights.cols)

    val outputRows = (inputRows - 1) * stride._1 - 2 * padding._1 + kernelRows
    val outputCols = (inputCols - 1) * stride._2 - 2 * padding._2 + kernelCols

    val output = DenseMatrix.zeros[Float](outputRows, outputCols)

    for (i <- 0 until inputRows) {
      for (j <- 0 until inputCols) {
        for (k <- 0 until kernelRows) {
          for (l <- 0 until kernelCols) {
            val row = i * stride._1 - padding._1 + k
            val col = j * stride._2 - padding._2 + l

            if (row >= 0 && row < outputRows && col >= 0 && col < outputCols) {
              output(row, col) += input(i, j) * weights(k, l)
            }
          }
        }
      }
    }

    output
  }

  def maxPooling(input: Array[Array[Float]], poolingSize: (Int, Int), stride: Int): (Array[Array[Float]],Array[(Int, Int)]) = {
    val (poolHeight, poolWidth) = poolingSize
    val numRows = (input.length )
    val numCols = (input(0).length )
    val poolingNumRows = ((input.length/poolWidth))
    val poolingNumCols = ((input(0).length/poolHeight))
    val result = Array.ofDim[Float](poolingNumRows, poolingNumCols)
    val maxIndices = new Array[(Int, Int)](poolingNumRows * poolingNumCols)
    var maxIndex = 0
    var maxRow = 0
    var maxCol = 0
    var maxRowIndex = 0
    var maxColIndex = 0

    for (r <- 0 until (numRows) by poolWidth) {
      maxCol = 0
      for (c <- 0 until (numCols) by poolHeight) {
        var maxVal = Float.MinValue
        for (i <- 0 until poolWidth) {
          for (j <- 0 until poolHeight) {
            val startRow = r * stride
            val startCol = c * stride

            val value = input(r + i)(c + j)
            if (value > maxVal) {
              maxVal = value
              maxRowIndex = startRow+i
              maxColIndex = startCol+j
            }
            if ((i + 1) == poolWidth && (j + 1) == poolHeight) {
              maxIndices(maxIndex) = (maxRowIndex, maxColIndex)
              maxIndex = maxIndex + 1
              result(maxRow)(maxCol) = maxVal
            }
          }
        }
        maxCol = maxCol + 1
      }
      maxRow = maxRow + 1
    }

    (result,maxIndices)
  }

  def avgPooling(input: Array[Array[Float]], poolingSize: (Int, Int), stride: Int): (Array[Array[Float]], Array[(Int, Int)]) = {
    val (poolHeight, poolWidth) = poolingSize
    val numRows = (input.length)
    val numCols = (input(0).length)
    val poolingNumRows = ((input.length / poolWidth))
    val poolingNumCols = ((input(0).length / poolHeight))
    val result = Array.ofDim[Float](poolingNumRows, poolingNumCols)

    val maxIndices = new Array[(Int, Int)](numRows * numCols)
    var maxIndex = 0
    var maxRow = 0
    var maxCol = 0
    var maxRowIndex = 0
    var maxColIndex = 0

    for (r <- 0 until (numRows) by poolWidth) {
      maxCol = 0
      for (c <- 0 until (numCols) by poolHeight) {
        var maxVal = 0.0f
        for (i <- 0 until poolWidth) {
          for (j <- 0 until poolHeight) {
            val startRow = r * stride
            val startCol = c * stride

            val value = input(r + i)(c + j)
            maxVal += value
            maxRowIndex = startRow + i
            maxColIndex = startCol + j
            maxIndices(maxIndex) = (maxRowIndex, maxColIndex)
            maxIndex = maxIndex + 1
            if ((i + 1) == poolWidth && (j + 1) == poolHeight) {
              result(maxRow)(maxCol) = maxVal/(poolDimCol+poolDimRow)
            }
          }
        }
        maxCol = maxCol + 1
      }
      maxRow = maxRow + 1
    }
    (result, maxIndices)
  }

  def pooling(convolutionFilter:ConvolutionalFilter, input: Array[Array[Float]]): (Array[Array[Float]], Array[(Int, Int)]) = {
    convolutionFilter.poolingStrategy match {
      case "max" =>
        maxPooling(input, (convolutionFilter.poolDimRow,convolutionFilter.poolDimCol), convolutionFilter.stride)
      case "avg" =>
        avgPooling(input, (convolutionFilter.poolDimRow, convolutionFilter.poolDimCol), convolutionFilter.stride)
    }
  }
}
