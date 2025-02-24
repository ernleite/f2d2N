package com.deeplearning

import akka.actor.typed.Behavior
import akka.actor.typed.receptionist.{Receptionist, ServiceKey}
import akka.actor.typed.scaladsl.{AbstractBehavior, ActorContext, Behaviors}
import com.deeplearning.ComputeEpochs.SetStats
import com.deeplearning.layer.{LayerFactory, WeightedLayer}

import scala.collection.immutable.HashMap
import scala.util.Random

trait ComputeWeightedSerializable
class Weighted(context: ActorContext[ComputeWeighted.WeightedCommand]) extends AbstractBehavior[ComputeWeighted.WeightedCommand](context) {
  import ComputeWeighted._
  private var layer:WeightedLayer = _
  private var eventFF: Int = 0
  private var eventBP: Int = 0

  override def onMessage(msg: ComputeWeighted.WeightedCommand): Behavior[ComputeWeighted.WeightedCommand] = {
    msg match {

      // Construct the inputs array sent by activation neurons on l-1 layer
      case Weights(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, activations: Array[Float], shards:Int, internalSubLayer: Int, layer: Int, params : scala.collection.mutable.HashMap[String,String]) =>
        eventFF = eventFF + 1

        if (this.layer == null) {
          this.layer = LayerFactory.getWeightedLayer(Network.getHiddenLayersType(layer, "weighted"))
        }
        this.layer.Weights(epoch,correlationId,yLabel,trainingCount,activations,shards, internalSubLayer,layer, params)
        this

      case BackPropagate(correlationId: String, delta: Array[Float],learningRate: Float, regularisation: Float, nInputs: Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int, params:scala.collection.mutable.HashMap[String,String]) =>
        eventBP = eventBP + 1

        this.layer.BackPropagate(correlationId, delta, learningRate, regularisation, nInputs, layer, internalSubLayer, fromInternalSubLayer, params,false)
        this

      case FeedForwardTest(correlationId: String, activations: Array[Float], internalSubLayer: Int, layer: Int) =>
        val nextLayer = layer + 1

        var hiddenLayerStep = 0
        var arraySize = 0

        if (LayerManager.IsLast(nextLayer)) {
          hiddenLayerStep = LayerManager.GetOutputLayerStep()
          arraySize = Network.OutputLayerDim
        }
        else {
          hiddenLayerStep = LayerManager.GetDenseWeightedLayerStep(layer)
          arraySize = Network.getHiddenLayersDim(layer, "weighted")
        }

        this.layer.FeedForwardTest(correlationId,activations, internalSubLayer,layer)
        this

      case getStats(replyTo:String, actorIndex : Int) =>
        val epoch = Network.EpochsRef(replyTo)
        epoch ! SetStats(eventFF,eventBP, "weightedLayer_"+ actorIndex)
        this

      case getGradientsNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, localIndex:Int) =>
        this.layer.getNeighbor(correlationId,gradients, layer,internalSubLayerCaller, localIndex)
        this

      case fromGradientsNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, localIndex:Int) =>
        this.layer.applyGradients(correlationId,gradients, layer,internalSubLayerCaller,localIndex)
        this

    }
  }
}

object ComputeWeighted {
  sealed trait WeightedCommand extends ComputeWeightedSerializable
  final case class Weights(Epoch:Int, CorrelationId: String, YLabel:Int, trainingCount:Int, Activations: Array[Float], Shards:Int, InternalSubLayer:Int, Layer:Int, Params : scala.collection.mutable.HashMap[String,String]) extends WeightedCommand
  final case class BackPropagate(CorrelationId: String, delta: Array[Float], LearningRate : Float, Regularisation:Float, nInputs:Int, Layer:Int, InternalSubLayer:Int, fromInternalSubLayer:Int, Params:scala.collection.mutable.HashMap[String,String]) extends WeightedCommand
  final case class FeedForwardTest(CorrelationId: String, Inputs: Array[Float], InternalSubLayer:Int, Layer:Int) extends WeightedCommand
  final case class getStats(replyTo: String, actorIndex : Int) extends WeightedCommand
  final case class getGradientsNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, localIndex:Int) extends WeightedCommand
  final case class fromGradientsNeighbor(correlationId: String, gradients: Array[Float], layer:Int,internalSubLayerCaller:Int, fromInternalSubLayer:Int) extends WeightedCommand


  def apply(actorId: String): Behavior[WeightedCommand] =
    Behaviors.setup { context =>
      context.system.receptionist ! Receptionist.Register(
        ServiceKey[ComputeWeighted.WeightedCommand](actorId), context.self
      )
      new Weighted(context)
    }

}