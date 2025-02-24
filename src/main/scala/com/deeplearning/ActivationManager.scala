package com.deeplearning

import breeze.linalg.{DenseMatrix, DenseVector, InjectNumericOps, max}
import breeze.numerics.{sigmoid, tanh}
import com.deeplearning.CostManager.dotProduct

object ActivationManager {
  def ComputeZ(ActivationFunction : String, z:Array[Float]) : Array[Float] = {
    var arr:Array[Float] = Array.ofDim[Float](10)
    arr = Array.fill[Float](z.length)(0)
    ActivationFunction match {
      case "SoftMax" =>
        /*
      var expTotal:Float = 0
        for (k:Int <- 0 until z.length) {
          expTotal += math.exp(z(k)).toFloat
        }
        for (k: Int <- 0 until z.length) {
          arr(k) = (math.exp(z(k))/expTotal).toFloat
        }
        arr
         */
        CostManager.softMax(z)
      case "SiLu" =>
        z * sigmoid(new DenseVector(z)).toArray
      case "Sigmoid" =>
        sigmoid(new DenseVector(z)).toArray
      case "Tanh" =>
        tanh(new DenseVector(z)).toArray
      case "Relu" =>
        z.map(x => if (x > 0) x else 0)
      case "LeakyRelu" =>
        val denseX = new DenseVector(z)
        val relu = denseX.map(x => if (x > 0) x else Network.LeakyReluAlpha * x)
        relu.toArray
    }
  }

  def ComputeZ(ActivationFunction: String, z:Float): Float = {
    ActivationFunction match {
      case "Relu" => if (z > 0) z else 0
      case "Tanh" => tanh(z)
      case "LeakyRelu" => max(z, z * Network.LeakyReluAlpha)
      case "SiLu" =>
        z*sigmoid(z)
      case "Sigmoid" =>
        sigmoid(z)
    }
  }

  def ComputePrime(ActivationFunction: String, z: Array[Float]): Array[Float] = {
    var arr: Array[Float] = Array.ofDim[Float](10)
    arr = Array.fill[Float](z.length)(0)

    ActivationFunction match {
      case "LeakyRelu" =>
        for (k: Int <- z.indices) {
          arr(k) = if (z(k) > 0.0) z(k) else Network.LeakyReluAlpha
        }
        arr
      case "Relu"=>
        for (k: Int <- z.indices) {
          arr(k) = if (z(k) > 0) z(k) else 0
        }
        arr
      case "Sigmoid" =>
        val mat = (1.0f - DenseMatrix(sigmoid(z)))
        val arr = dotProduct(sigmoid(z),mat.toArray)
        arr
      case "SiLu"  =>
        val mat = (1.0f - DenseMatrix(sigmoid(z)))
        val arr = dotProduct(z*sigmoid(z),mat.toArray)
        arr
      case "Tanh" =>
        val tanhX = tanh(new DenseVector(z)).toArray
        (1.0f - DenseMatrix(dotProduct(tanhX, tanhX))).toArray
    }
  }
}
