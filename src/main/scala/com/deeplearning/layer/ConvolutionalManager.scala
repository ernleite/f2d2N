package com.deeplearning.layer

import breeze.linalg.DenseMatrix

object ConvolutionalManager {

  def deconvolution(input: DenseMatrix[Float], kernel: DenseMatrix[Float]): DenseMatrix[Float] = {
    // Get dimensions of the input and kernel
    val (inputHeight, inputWidth) = (input.rows, input.cols)
    val (kernelHeight, kernelWidth) = (kernel.rows, kernel.cols)

    // Calculate the output size
    val outputHeight = inputHeight + kernelHeight - 1
    val outputWidth = inputWidth + kernelWidth - 1

    // Create a new matrix to hold the deconvolution result
    val deconvResult = DenseMatrix.zeros[Float](outputHeight, outputWidth)

    // Perform the deconvolution (transposed convolution) operation
    for (r <- 0 until inputHeight) {
      for (c <- 0 until inputWidth) {
        val inputSlice = input(r, c)
        val outputSlice = deconvResult(r until r + kernelHeight, c until c + kernelWidth)
        outputSlice :+= (inputSlice * kernel)
      }
    }

    deconvResult
  }

  def convolution2D(input: Array[Array[Float]], filterArray: Array[Float]): (Array[Array[Float]], Array[Array[Float]]) = {
    // Parse the filter string and create a filter array of Float values


    // Get the dimensions of the input array and filter
    val inputRows = input.length
    val inputCols = input(0).length
    val filterSize = math.sqrt(filterArray.length).toInt

    // Calculate the output array dimensions
    val outputRows = inputRows - filterSize + 1
    val outputCols = inputCols - filterSize + 1

    var outputMatrix: Array[Array[Float]] = Array[Array[Float]]()
    outputMatrix = Array.ofDim(outputRows)

    // Initialize the output array with zeros
    var output = ""
    val deltaMatrix:Array[Array[Float]] = Array.tabulate(outputRows * outputCols,filterArray.length) {
      (_, _) => 0.0f
    }

    // Convolution operation
    var deltaMatrixIndex = 0
    for (i <- 0 until outputRows) {
      outputMatrix(i) = Array.ofDim(outputCols)
      for (j <- 0 until outputCols) {
        var index = 0
        for (k <- 0 until filterSize) {
          for (l <- 0 until filterSize) {
            // Calculate the index in the input array
            val inputRowIndex = i + k
            val inputColIndex = j + l

            // Calculate the index in the filter array
            val filterIndex = k * filterSize + l

            // Calculate the index in the output array
            //val outputIndex = i * outputCols + j

            // Apply the convolution operation and update the output array
            //val t = inputRowIndex.toString + " " + inputColIndex.toString + " " + filterIndex.toString + " " + input(inputRowIndex)(inputColIndex) + ";"
            deltaMatrix(deltaMatrixIndex)(index) = input(inputRowIndex)(inputColIndex)
            outputMatrix(i)(j) += input(inputRowIndex)(inputColIndex) * filterArray(filterIndex)
            //output = inputRowIndex.toString + " " + inputColIndex.toString + " " + filterIndex.toString + " " + input(inputRowIndex)(inputColIndex) + ";"
            index = index + 1
          }
        }
        deltaMatrixIndex = deltaMatrixIndex + 1
      }
    }
    (outputMatrix, deltaMatrix)
  }

}
