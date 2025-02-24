package com.deeplearning

object DerivativeManager {

  def softmaxDerivative(output: Array[Float]): Array[Array[Float]] = {
    val n = output.length
    val result = Array.ofDim[Float](n, n)
    for (i <- 0 until n) {
      for (j <- 0 until n) {
        if (i == j) {
          result(i)(j) = output(i) * (1 - output(i))
        } else {
          result(i)(j) = -output(i) * output(j)
        }
      }
    }
    result
  }

  def reluDerivative(x: Float): Float = {
    if (x > 0) 1 else 0
  }

}
