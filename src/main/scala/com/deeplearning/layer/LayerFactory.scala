package com.deeplearning.layer

import com.deeplearning.Network

object LayerFactory {
  def getInputLayer(layerType:String): InputLayer = {
    if (layerType == "Dense") {
      new DenseInputLayer()
    } else if (layerType == "Conv2d") {
      new ConvolutionInputLayer()
    }
    else
      new DenseInputLayer()
  }

  def getActivationLayer(layer: Int): ActivationLayer = {
    // inputlayer
    if (layer == 0) {
      if (Network.InputLayerType == "Dense") {
        new DenseActivationLayer()
      }
      else if (Network.InputLayerType == "Conv2d") {
        new ConvolutionActivationLayer()
      }
      else new DenseActivationLayer()
    }
    else {
      Network.getActivationLayersType(layer) match  {
        case "Conv2d" =>
          new ConvolutionActivationLayer()
        case _ =>
          new DenseActivationLayer()
      }
    }
  }

  def getWeightedLayer(layerType: String): WeightedLayer = {
    if (layerType == "Dense") {
      new DenseWeightedLayer()
    }
    else if (layerType == "Conv2d") {
      new ConvolutionWeightedLayer()
    }
    else if (layerType == "Flatten") {
      new FlattenWeightedLayer()
    }
    else
      new DenseWeightedLayer()
  }

  def maxPooling(input: Array[Array[Float]], poolSize: Int): Array[Array[Float]] = {
    val numRows = input.length
    val numCols = input(0).length

    val pooledRows = numRows / poolSize
    val pooledCols = numCols / poolSize

    val output = Array.ofDim[Float](pooledRows, pooledCols)

    for (i <- 0 until pooledRows) {
      for (j <- 0 until pooledCols) {
        val startRow = i * poolSize
        val startCol = j * poolSize

        var maxVal = Float.MinValue
        for (row <- startRow until startRow + poolSize) {
          for (col <- startCol until startCol + poolSize) {
            if (input(row)(col) > maxVal) {
              maxVal = input(row)(col)
            }
          }
        }
        output(i)(j) = maxVal
      }
    }
    output
  }


  def convolution2D(input: Array[Array[Float]], filter: Array[Array[Float]], stride: Int): Array[Array[Float]]
  = {
    val inputRows = input.length
    val inputCols = input(0).length
    val filterRows = filter.length
    val filterCols = filter(0).length
    val outputRows = (inputRows - filterRows) / stride + 1
    val outputCols = (inputCols - filterCols) / stride + 1

    val output = Array.ofDim[Float](outputRows, outputCols)
    for (i <- 0 until outputRows; j <- 0 until outputCols) {
      var sum = 0.0f
      for (k <- 0 until filterRows; l <- 0 until filterCols) {
        sum += input(i * stride + k)(j * stride + l) * filter(k)(l)
      }
      output(i)(j) = sum
    }

    output
  }

}
