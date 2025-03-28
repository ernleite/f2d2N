package com.deeplearning.layer

import breeze.linalg._
import breeze.numerics._
import breeze.stats.{mean => breezeMean, variance => breezeVariance}

import breeze.linalg._
import breeze.numerics._
import breeze.stats.{mean => breezeMean, variance => breezeVariance}

object LayerNorm {
  def apply(normalizedShape: Int, eps: Double = 1e-5, elementwiseAffine: Boolean = true): LayerNorm =
    new LayerNorm(normalizedShape, eps, elementwiseAffine)
  import scala.math.exp

  def clipByNorm(gradients: Array[Float], maxNorm: Float = 1.0f, epsilon: Float = 1e-7f): Array[Float] = {
    var normSquared = 0.0f
    var i = 0
    while (i < gradients.length) {
      normSquared += gradients(i) * gradients(i)
      i += 1
    }
    val norm = math.sqrt(normSquared).toFloat
    if (norm <= maxNorm + epsilon) gradients
    else {
      val scale = maxNorm / (norm + epsilon)
      val clipped = new Array[Float](gradients.length)
      i = 0
      while (i < gradients.length) {
        clipped(i) = gradients(i) * scale
        i += 1
      }
      clipped
    }
  }


  def stabilizedSoftmax(z: Array[Float], temperature: Float = 1.0f): Array[Float] = {
    if (z.isEmpty) return Array.empty[Float]

    // 1. Input normalization (scale to [-1, 1] range)
    val maxAbs = z.map(_.abs).max.max(1e-6f) // Prevent division by zero
    val normalized = z.map(_ / maxAbs)        // Now in [-1, 1] range [1][5]

    // 2. Temperature scaling and max subtraction
    val scaled = normalized.map(_ / temperature)
    val maxVal = scaled.max
    val exps = scaled.map(x => exp(x - maxVal).toFloat) // Numerical stability [3][7]

    // 3. Probability normalization
    val sumExps = exps.sum.max(1e-6f) // Avoid division by zero
    exps.map(_ / sumExps)
  }
}

class LayerNorm private[LayerNorm] (normalizedShape: Int, eps: Double, elementwiseAffine: Boolean) {
  private val weight = if (elementwiseAffine) DenseVector.ones[Double](normalizedShape) else null
  private val bias = if (elementwiseAffine) DenseVector.zeros[Double](normalizedShape) else null

  def forward(input: DenseMatrix[Double]): DenseMatrix[Double] = {
    val meanVector = breezeMean(input(::, *))
    val varianceVector = breezeVariance(input(::, *))

    val normalized = DenseMatrix.tabulate(input.rows, input.cols) { (i, j) =>
      (input(i, j) - meanVector(j)) / math.sqrt(varianceVector(j) + eps)
    }

    if (elementwiseAffine) {
      DenseMatrix.tabulate(normalized.rows, normalized.cols) { (i, j) =>
        normalized(i, j) * weight(j) + bias(j)
      }
    } else {
      normalized
    }
  }
}