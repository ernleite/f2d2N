package com.deeplearning


/*
object AdamOptimizer {

  def optimize(
                weights: List[DenseMatrix[Float]],
                gradients: List[DenseMatrix[Float]],
                learningRate: Float = 0.001f,
                beta1: Float = 0.9f,
                beta2: Float = 0.999f,
                epsilon: Double = 1e-8,
                t: Int = 1
              ): List[DenseMatrix[Float]] = {
    val m = weights.map(w => DenseMatrix.zeros[Float](w.rows, w.cols))
    val v = weights.map(w => DenseMatrix.zeros[Float](w.rows, w.cols))
    val beta1t = Math.pow(beta1, t)
    val beta2t = Math.pow(beta2, t)

    // Update m and v
    for (i <- weights.indices) {
      m(i) := beta1 * m(i) + (1 - beta1) * gradients(i)
      v(i) := beta2 * v(i) + (1 - beta2) * (gradients(i) *:* gradients(i))
    }

    // Compute bias-corrected first and second moment estimates
    val mc = m.map(w => w / (1 - beta1t).toFloat)
    val vc = v.map(w => w / (1 - beta2t).toFloat)

    // Update weights
    val updatedWeights = for (i <- weights.indices) yield {
      //val deltaW = -learningRate * (mc(i) /:/ (vc(i).map(Math.sqrt) + epsilon))
      //weights(i) + deltaW
    }

    updatedWeights.toList
  }

}
 */
