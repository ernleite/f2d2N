package com.deeplearning

import math.sqrt
import breeze.stats._
import breeze.linalg._

import scala.collection.immutable.HashMap
object Normalisation {

  def flattenHashMap(inputs:scala.collection.mutable.HashMap[String,Array[Float]]) : Array[Float] = {
    inputs.values.flatten.toArray
  }

  def getMean(input: Array[Float]): Float = {
    val dense = DenseVector(input)
    mean(dense)
  }

  def getVariance(input: Array[Float]): Float = {
    val dense = DenseVector(input)
    variance(dense).toFloat
  }

  def getMeanAndVariance(input:Array[Float]) : (Float,Float,Float, Int) = {
    val dense = DenseVector(input)
    val tuple = meanAndVariance(dense)
    (tuple.mean.toFloat, tuple.variance.toFloat, tuple.stdDev.toFloat,tuple.count.toInt )
  }

  def batchNormalize(input: Array[Float], mean: Float, std: Float, gamma: Float, beta: Float): Array[Float] = {
    val epsilon = 1e-5f // Small constant for numerical stability
    // Convert the input array to a DenseVector
    val inputVector = DenseVector(input)
    // Calculate normalized and scaled output using batch normalization formula
    ((gamma * ((inputVector - mean) / (std + epsilon))) + beta).toArray
  }
}