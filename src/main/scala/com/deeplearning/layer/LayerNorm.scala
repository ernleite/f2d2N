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