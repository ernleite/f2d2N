package com.deeplearning
import ai.djl.Device
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.training.loss.Loss
import breeze.numerics.{sigmoid, sqrt}
import ai.djl.nn.Activation


object CostManager {
  val scale = 1.0507f
  val alpha = 1.67326f

  def batchNormalize(input: Array[Float], epsilon: Float = 1e-5f): Array[Float] = {
    val mean = input.sum / input.length.toFloat
    val variance = input.map(x => (x - mean) * (x - mean)).sum / input.length.toFloat
    val normalized = input.map(x => (x - mean) / sqrt(variance + epsilon))
    normalized
  }

  def Compute(CostFunction: String, y:Array[Float], a: Array[Float]): Float = {
    CostFunction match {
      case "Quadratic" =>
        var c: Float = 0.0f
        for (i <- 0 until y.length) {
          val z = y(i)-a(i)
          val x = math.pow(z, 2).toFloat
          c += x
        }
        c / 2
      case "SSE" =>
        var c: Float = 0.0f
        for (i <- 0 until y.length) {
          val z = y(i) - a(i)
          val x = math.pow(z, 2).toFloat
          c += x
        }
        c / 2
    }
  }

  def Delta(trueLabels: Array[Float], prediction: Array[Float]): Array[Float] = {
    CostManager.minus(prediction,trueLabels)
  }

  def meanSquareError(predictions: Array[Float], actualValues: Array[Float]): Float = {
    val squaredErrors = predictions.zip(actualValues).map { case (prediction, actualValue) =>
      val error = prediction - actualValue
      error * error
    }
    squaredErrors.sum / squaredErrors.length
  }

  def EliminateNaN(arr: Array[Float]): Array[Float] = {
    arr.map {
      case x if x.isNaN => 0.0f
      case x => x
    }
  }
  import scala.math.log

  def categoricalCrossEntropy3(trueLabels: Array[Float], prediction: Array[Float], size:Int, scalar:Float, nablas: Array[Float], bias: Array[Float]): (Float, Array[Float]) = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val pred: NDArray = manager.create(prediction).reshape(trueLabels.size, size)
    val trueLab : NDArray = manager.create(trueLabels).reshape(trueLabels.size, 1)
    val nabla : NDArray = manager.create(nablas).reshape(nablas.size/size, size).sum(Array(0))
    val biases : NDArray = manager.create(bias)
    val ndList1: NDList = new NDList()
    ndList1.add(trueLab)
    val ndList2: NDList = new NDList()
    ndList2.add(pred)
    // Calculate the softmax cross-entropy loss for the minibatch
    val lossOutput = Loss.softmaxCrossEntropyLoss().evaluate(ndList1 , ndList2).toFloatArray

    val tmp2 = nabla.mul(scalar)
    val tmp3  = biases.subi(tmp2).toFloatArray

    ndList1.close()
    ndList2.close()
    pred.close()
    trueLab.close()
    // Don't forget to close the manager when done
    manager.close()
    (lossOutput(0),tmp3)
  }

  def applyBias(size:Int, scalar:Float, nablas: Array[Float], bias: Array[Float]): (Array[Float]) = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val nabla : NDArray = manager.create(nablas).reshape(nablas.size/size, size).sum(Array(0))
    val biases : NDArray = manager.create(bias)
    val tmp2 = nabla.mul(scalar)
    val tmp3  = biases.subi(tmp2).toFloatArray
    nabla.close()
    biases.close()
    manager.close()
    tmp3
  }

  def layerNorm(input: Array[Float], epsilon: Float = 1e-5f): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val x: NDArray = manager.create(input)
    val shape = x.getShape
    val lastDim = shape.dimension() - 1

    // Calculate mean and variance along the last dimension
    val mean = x.mean(Array(lastDim), true)
    val variance = x.sub(mean).pow(2).mean(Array(lastDim), true)

    // Normalize
    val xNorm = x.sub(mean).div(variance.add(epsilon).sqrt())

    // In a full implementation, you would apply gamma and beta here
    // For simplicity, we're omitting these learnable parameters
    // xNorm = xNorm.mul(gamma).add(beta)

    val c = xNorm.toFloatArray
    manager.close()
    c
  }


  def categoricalCrossEntropy2(trueLabels: Array[Array[Float]], prediction: Array[Array[Float]]): Float = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())

    //if (Network.GpuMode) manager = NDManager.newBaseManager(Device.gpu(0))

    val pred: NDArray = manager.create(prediction)
    val trueLab : NDArray = manager.create(trueLabels)
    val ndList1: NDList = new NDList()
    ndList1.add(trueLab)
    val ndList2: NDList = new NDList()
    ndList2.add(pred)
    // Calculate the softmax cross-entropy loss for the minibatch
    val lossOutput = Loss.softmaxCrossEntropyLoss().evaluate(ndList1 , ndList2).toFloatArray
    ndList1.close()
    ndList2.close()
    pred.close()
    trueLab.close()
    // Don't forget to close the manager when done
    manager.close()
    lossOutput(0)
  }

  def normalization(arr: Array[Float]): Array[Float] = {
    val max = arr.max
    val min = arr.min
    arr.map(v => (v - min) / (max - min))
  }

  def ComputePrime(ActivationFunction: String, z: Array[Float]): Array[Float] = {
      val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
      val zArray : NDArray = manager.create(z)
      var c = Array[Float]()

      ActivationFunction match {
        case "Sigmoid" =>
          val sigmoidZ = Activation.sigmoid(zArray)
          val mat = manager.ones(zArray.getShape).sub(sigmoidZ)
          c = sigmoidZ.mul(mat).toFloatArray
        case "Relu" =>
          val ones = manager.ones(zArray.getShape)
          c = zArray.gt(0).toType(DataType.FLOAT32, false).mul(ones).toFloatArray
        case "LeakyRelu" =>
          val positiveGradient = zArray.gte(0).toType(DataType.FLOAT32, false)
          val negativeGradient = zArray.lt(0).toType(DataType.FLOAT32, false).mul(Network.LeakyReluAlpha)
          c = positiveGradient.add(negativeGradient).toFloatArray
        case "SiLu" =>
          val sigmoidX = Activation.sigmoid(zArray)
          val siluX = zArray.mul(sigmoidX)
          c = sigmoidX.add(siluX.mul(zArray.getManager.ones(zArray.getShape).sub(sigmoidX))).toFloatArray
        case "ELu" =>
          val condition = zArray.gt(0)
          val positiveValues = manager.ones(zArray.getShape)
          val negativeValues = zArray.mul(alpha).add(alpha)
          c = condition.mul(positiveValues).add(condition.logicalNot().mul(negativeValues)).toFloatArray
        case "SeLu" =>
          val condition = zArray.gt(0)
          val positiveValues = manager.ones(zArray.getShape).mul(scale)
          val negativeValues = zArray.mul(scale).mul(alpha).exp()
          c = condition.mul(positiveValues).add(condition.logicalNot().mul(negativeValues)).toFloatArray
      }
      zArray.close()
      manager.close()
      c
  }

  def ComputeZ(ActivationFunction : String, z:Array[Float]) : Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val zArray : NDArray = manager.create(z)
    var c = Array[Float]()
    ActivationFunction match {
      case "Sigmoid" =>
        c = Activation.sigmoid(zArray).toFloatArray
      case "Relu" =>
        c = Activation.relu(zArray).toFloatArray
      case "LeakyRelu" =>
        c = Activation.leakyRelu(zArray, Network.LeakyReluAlpha).toFloatArray
      case "SiLu" =>
        val sigmoidX = Activation.sigmoid(zArray)
        // Calculate SiLU(x) = x * sigmoid(x)
        c = zArray.mul(sigmoidX).toFloatArray
      case "ELu" =>
        c = Activation.elu(zArray, alpha).toFloatArray
      case "SeLu" =>
        c = Activation.selu(zArray).toFloatArray
    }
    manager.close()
    zArray.close()
    c
  }


  def categoricalCrossEntropy(trueLabels: Array[Float], prediction: Array[Float]): Float = {
      val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
      // Generate example prediction and target tensors (replace with your actual data)
      val array1 = manager.create(trueLabels)
      val fromMat1: NDArray = manager.from(array1)
      val fromList: NDList = new NDList()
      fromList.add(fromMat1)
      val array2 = manager.create(prediction)
      val fromMat2: NDArray = manager.from(array2)
      val fromPred: NDList = new NDList()
      fromPred.add(fromMat2)

      // Calculate the categorical cross-entropy loss
      val loss = Loss.softmaxCrossEntropyLoss().evaluate(fromList, fromPred).toFloatArray
      fromMat1.close()
      fromPred.close()
      fromList.close()
      fromMat2.close()
      manager.close()
      loss(0)
  }

 def crossEntropyLoss(predictedProbabilities: Array[Float], trueLabels: Array[Float]): Float = {
    require(predictedProbabilities.length == trueLabels.length, "Input arrays must have the same length.")

    val numClasses = predictedProbabilities.length
    var loss = 0.0f

    for (i <- 0 until numClasses) {
      // Avoid taking the log of 0 by adding a small epsilon
      val epsilon = 1e-15f
      val probability = Math.max(predictedProbabilities(i), epsilon)

      loss += trueLabels(i) * log(probability).toFloat
    }

    -loss
  }

  def replaceNaN(v: Array[Float], value: Float): Array[Float] = {
    v.map(x => if (x.isNaN) value else x)
  }

  def getWeightedSum(arr: Array[Float], l: Int): Array[Float] = {
    // Create an NDManager
    if (Network.GpuMode) {
      val manager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      // Convert the input array to an NDArray
      val ndArr = manager.create(arr)

      // Reshape the array to group elements
      val rows = arr.length / l
      val reshaped = ndArr.reshape(rows, l)

      // Sum along the second axis (axis 1)
      val summed = reshaped.sum(Array(1))

      // Convert the result back to a Scala array
      val c = summed.toFloatArray
     // Close the manager to release resources
      manager.close()
      c
    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      // Convert the input array to an NDArray
      val ndArr = manager.create(arr)

      // Reshape the array to group elements
      val rows = arr.length / l
      val reshaped = ndArr.reshape(rows, l)

      // Sum along the second axis (axis 1)
      val summed = reshaped.sum(Array(1))

      // Convert the result back to a Scala array
      val c  = summed.toFloatArray
      // Close the manager to release resources
      manager.close()
      c
    }
  }


  def dotProduct(mat1 : Array[Float], mat2: Array[Float]) : Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)
    val c = fromMat1.mul(fromMat2).toFloatArray
    fromMat1.close()
    fromMat1.close()
    array1.close()
    array2.close()
    manager.close()
    c
  }

  def dotProduct3(size: Int, mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    var output = Array[Array[Float]]()
    output = Array.ofDim(mat1.length)
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)
    val matrixA2D = fromMat1.reshape(mat1.size, 1)
    val matrixB2D = fromMat2.reshape(1, mat2.size)

    val result =  matrixA2D.matMul(matrixB2D).toFloatArray
    fromMat1.close()
    fromMat2.close()
    array1.close()
    array2.close()
    manager.close()

    result
  }

  def matMulScalar(scalar: Float, input: Array[Float]): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(input)
    val c2 = array1.mul(scalar)
    array1.close()
    val tmp = c2.toFloatArray
    c2.close()
    manager.close()
    tmp
  }

  def applyGradients(mat1: Array[Float], mat2: Array[Float], outputSize:Int, scalar1: Float, scalar2:Float, mat3:Array[Float]): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val array3 = manager.create(mat3)

    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)
    val fromMat3: NDArray = manager.from(array3)

    val mat1Reshaped = fromMat1.reshape(outputSize,mat1.size/ outputSize, 1)
    val mat2Reshaped = fromMat2.reshape(outputSize, 1, mat2.size/outputSize)

    // Perform batch matrix multiplication
    val result = mat1Reshaped.matMul(mat2Reshaped)
    val summedResult = result.sum(Array(0)).reshape(-1)
    val tmp2 = summedResult.mul(scalar1)
    val tmp1 = fromMat3.mul(scalar2)
    val c = tmp1.sub(tmp2).toFloatArray
    mat1Reshaped.close()
    mat2Reshaped.close()
    fromMat3.close()
    fromMat1.close()
    fromMat2.close()
    manager.close()
    c
  }

  def getRange(weights: Int, from: Int, to:Int, layer: Int): Array[Int] = {
    val size = from * to
    val starting2 = weights*layer
    val fromMul = starting2/from
    val residu = starting2%from
    val weigthSlide = size/weights
    var residu1 = 0
    var residu2 = 0
    if (residu == 0) {
      val endWeightIndex = starting2 + weights
      val startIndex2 = from * fromMul
      residu1 = starting2-startIndex2
      val test = (endWeightIndex)%from
      residu2 = from-test
    }
    else {
      val endWeightIndex = starting2 + weights
      val startIndex2 = from * fromMul
      residu1 = starting2-startIndex2
      val test = (endWeightIndex)%from
      if (test != 0)
        residu2 = from-test
    }

    Array(residu1, residu2)
  }

  def getIndex(from:Int, to:Int, position:Int) : Int= {
    val divide = (to/from.toFloat).toFloat
    val pos = divide*(position)
    pos.toInt
  }

  def layerNormalization(input: Array[Float], epsilon: Float = 1e-5f): Array[Float] = {
    // Create an NDManager to manage memory
    val manager = NDManager.newBaseManager()

    try {
      // Convert input Array[Float] to NDArray
      val inputNDArray = manager.create(input)

      // Calculate mean
      val mean = inputNDArray.mean()

      // Calculate variance
      val variance = inputNDArray.sub(mean).pow(2).mean()

      // Normalize the input
      val normalized = inputNDArray.sub(mean).div(variance.add(epsilon).sqrt())

      // Convert the result back to Array[Float]
      normalized.toFloatArray
    } finally {
      // Close the NDManager to release resources
      manager.close()
    }
  }

  def applyGradientsLight(mat1: Array[Float], mat2: Array[Float], outputSize:Int, scalar1: Float, scalar2:Float): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)

    val fromMat1: NDArray = manager.from(array1)
    val mat1Reshaped = fromMat1.reshape(1, 1, mat1.size)
    val fromMat2: NDArray = manager.from(array2)

    val tmp2 = mat1Reshaped.mul(scalar1)
    val test = tmp2.toFloatArray
    val tmp1 = fromMat2.mul(scalar2)
    val c = tmp1.sub(tmp2).toFloatArray

    fromMat1.close()
    fromMat2.close()
    manager.close()
    c
  }

  def roundUpIfFractional(d: Double): Int = {
    val intPart = d.toInt
    if (d > intPart) intPart + 1 else intPart
  }

  def dotInputs(mat1: Array[Float], mat2: Array[Float], outputSize:Int): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)
    val weights = fromMat1.reshape(mat2.size,outputSize)
    val c= fromMat2.transpose().matMul(weights).toFloatArray
    array1.close()
    array2.close()
    fromMat1.close()
    fromMat2.close()
    weights.close()
    manager.close()
    c
  }

  def initInputs(mat1: Array[Float], mat2: Array[Float], outputSize:Int): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)
    val weights = fromMat1.reshape(outputSize,mat2.size)
    // 3. Perform linear transformation
    val c=weights.matMul(fromMat2).toFloatArray

    array1.close()
    array2.close()
    fromMat1.close()
    fromMat2.close()
    weights.close()
    manager.close
    c
  }


  def matMul2(mat1:Array[Array[Float]], mat2:Array[Float], size1: Int, size2:Int) : Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)

    // Perform the multiplication
    // Convert Scala arrays to NDArrays
    val ndArray1 = manager.from(array1).reshape(size2, size1)
    val ndArray2 = manager.from(array2).reshape(size2, 1)

    // Perform the multiplication and sum the results
    val result = ndArray1.mul(ndArray2)

    fromMat2.close()
    fromMat1.close()
    array1.close()
    array2.close()
    val tmp =  result.toFloatArray
    result.close()
    manager.close()
    tmp
  }

  def flattenSum(mat1:Array[Array[Float]]) : Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val fromMat1: NDArray = manager.from(array1)
    val c2 = fromMat1.sum(Array(0))
    fromMat1.close()
    array1.close()
    val tmp = c2.toFloatArray
    c2.close()
    manager.close()
    tmp
  }


  def matMul(mat1:Array[Array[Float]], mat2:Array[Float]) : Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)
    val c2 = fromMat2.matMul(fromMat1)
    fromMat2.close()
    fromMat1.close()
    array1.close()
    array2.close()
    val tmp =  c2.toFloatArray
    c2.close()
    manager.close()
    tmp
  }

  def dotProduct4(mat1: Array[Array[Float]], mat2: Array[Float]): Array[Float] = {
    val bT = mat1.transpose
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(bT)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)

    val c = fromMat1.matMul(fromMat2).toFloatArray
    fromMat2.close()
    fromMat1.close()
    array1.close()
    array2.close()
    manager.close()
    c
  }

  def dotProduct6(size: Int, mat1: Array[Array[Float]], mat2: Array[Float]): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)

    val c2 = fromMat1.matMul(fromMat2).toFloatArray
    fromMat2.close()
    fromMat1.close()
    array1.close()
    array2.close()
    manager.close()
    c2
  }

  def minusScalar(mat1: Array[Float], scalar: Float): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val fromMat1: NDArray = manager.from(array1)

    val c = fromMat1.subi(scalar).toFloatArray
    fromMat1.close()
    array1.close()
    manager.close()
    c
  }


  def minus(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)

    val c = fromMat1.subi(fromMat2).toFloatArray
    fromMat1.close()
    fromMat2.close()
    array1.close()
    array2.close()
    manager.close()
    c
  }

  def sum(mat1 : Array[Float], scalar: Float): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val fromMat1: NDArray = manager.from(array1)

    val c = fromMat1.addi(scalar).toFloatArray
    fromMat1.close()
    array1.close()
    manager.close()
    c
  }

  def divide(mat1: Array[Float], scalar: Float): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val fromMat1: NDArray = manager.from(array1)

    val c = fromMat1.divi(scalar).toFloatArray
    fromMat1.close()
    array1.close()
    manager.close()
    c
  }

  def sum2(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)

    val c = fromMat1.addi(fromMat2).toFloatArray
    fromMat1.close()
    fromMat2.close()
    array1.close()
    array2.close()

    manager.close()
    c
  }

  def scalling(array: Array[Float], min: Float, max: Float, rangeX: Float, rangeY: Float): Array[Float] = {
    val scale = (rangeY - rangeX) / (max - min)
    val b = rangeX - scale * min
    array.map(_ * scale + b)
  }


  def softMax(mat:Array[Float]) : Array[Float] = {
    val manager: NDManager = if(Network.GpuMode) NDManager.newBaseManager(Device.gpu(0)) else NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat)
    val fromMat1: NDArray = manager.from(array1)

    val c = fromMat1.softmax(0).toFloatArray
    fromMat1.close()
    array1.close()
    manager.close()
    c
  }
}
