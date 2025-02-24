package com.deeplearning

object LayerManager {

  def GetLayerStep(LayerIndex: Int, layer:String) : Int = {
    if (LayerIndex == 0) {
      val split = Network.InputLayerDim
      val size = Network.InputLayer
      size / split
    }
    else {
      val split = Network.getHiddenLayersDim(LayerIndex, layer)
      val size = Network.HiddenLayers(LayerIndex - 1)
      size / split
    }
  }

  def calculateResidue(originalArray: Int, step: Int): Int = {
    val startIndex = originalArray % 15
    startIndex
  }

  def GetDenseInputLayerStep(): Int = {
      val split = Network.InputLayerDim
      val size = Network.InputLayer
      size / split
  }
  def GetDenseActivationLayerStep(LayerIndex : Int) : Int = {
    var indx = 0
    if  (LayerIndex>1) indx = (LayerIndex-1)/2
    val split = Network.HiddenLayersDim(indx)
    val size = Network.HiddenLayers(indx)
    //size / split
    size
  }

  def GetDenseWeightedLayerStep(LayerIndex: Int): Int = {
    val indx = (LayerIndex / 2) -1
    val split = Network.HiddenLayersDim(indx)
    val size = Network.HiddenLayers(indx)
    size / split
  }


  def GetHiddenLayerStep(LayerIndex: Int, layer:String): Int = {
    if (LayerIndex == 0) {
      val split = Network.InputLayerDim
      val size = Network.InputLayer
      size / split
    }
    else {
      val split = Network.getHiddenLayersDim(LayerIndex, layer)
      val size = Network.HiddenLayers(LayerIndex - 1)
      size / split
    }
  }

  def GetOutputLayerStep(): Int = {
    val split = Network.OutputLayerDim
    val size = Network.OutputLayer
    size / split
  }

  def GetInputLayerStep(): Int = {
    val split = Network.InputLayerDim
    val size = Network.InputLayer
    size / split
  }

  def IsLast(LayerIndex : Int) : Boolean = {
    val test = Network.HiddenLayers.length
    ((LayerIndex) > Network.HiddenLayers.length*2)
  }

  def IsFirst(LayerIndex: Int): Boolean = {
    (LayerIndex == 0)
  }

  def IsFirstConvolution(LayerIndex: Int): Boolean = {
    (LayerIndex == 1)
  }

}
