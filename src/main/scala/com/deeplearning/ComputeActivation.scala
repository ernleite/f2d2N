package com.deeplearning

import akka.actor.typed.Behavior
import akka.actor.typed.receptionist.{Receptionist, ServiceKey}
import akka.actor.typed.scaladsl.{AbstractBehavior, ActorContext, Behaviors}
import com.deeplearning.ComputeEpochs.SetStats
import com.deeplearning.layer.{ActivationLayer, LayerFactory}
class Activation(context: ActorContext[ComputeActivation.ActivationCommand]) extends AbstractBehavior[ComputeActivation.ActivationCommand](context) {
  import ComputeActivation._

  private var layer:ActivationLayer = _
  private var eventFF:Int = 0
  private var eventBP:Int = 0

  override def onMessage(msg: ComputeActivation.ActivationCommand): Behavior[ComputeActivation.ActivationCommand] = {
    msg match {
      case ComputeZ(epoch:Int, correlationId: String, yLabel:Int, trainingCount:Int, shardedWeighted: Array[Float], internalSubLayer:Int,fromInternalSubLayer:Int,  layer:Int, shards: Int, params:scala.collection.mutable.HashMap[String,String], weights: Array[Float]) => {
        eventFF = eventFF + 1

        if (this.layer == null) {
          this.layer = LayerFactory.getActivationLayer(layer)
        }
        this.layer.ComputeZ(epoch,correlationId, yLabel, trainingCount, shardedWeighted, internalSubLayer, fromInternalSubLayer, layer, shards, params)
      }
      this

      case BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, shards:Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer:Int, params : scala.collection.mutable.HashMap[String,String]) =>
        eventBP = eventBP + 1
        this.layer.BackPropagate(correlationId, delta, learningRate, regularisation, nInputs, shards, layer, internalSubLayer, fromInternalSubLayer, params)
        this

      case FeedForwardTest(correlationId: String, shardedWeighted: Array[Float], internalSubLayer: Int, fromInternalSubLayer:Int, layer: Int, shards: Int) =>
        // bias initialized only one time during the training cycle
        this.layer.FeedForwardTest(correlationId,shardedWeighted,internalSubLayer,fromInternalSubLayer,layer, shards)
        this

      case getStats(replyTo:String, actorIndex : Int) =>
        val epoch = Network.EpochsRef(replyTo)
        epoch ! SetStats(eventFF,eventBP, "activationLayer_" + actorIndex)
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
trait ComputeActivationSerializable
object ComputeActivation {
  sealed trait ActivationCommand extends ComputeActivationSerializable
  final case class ComputeZ(Epoch:Int, CorrelationId: String, yLabel:Int, trainingCount:Int, Weighted: Array[Float], InternalSubLayer:Int, FromInternalSubLayer:Int, Layer:Int, Shards: Int, Params:scala.collection.mutable.HashMap[String,String], Weights: Array[Float]) extends ActivationCommand
  final case class BackPropagate(CorrelationId: String, delta: Array[Float], learningRate : Float, regularisation:Float, nInputs:Int, Shards:Int, Layer:Int, InternalSubLayer:Int, fromInternalSubLayer:Int, params : scala.collection.mutable.HashMap[String,String]) extends ActivationCommand
  final case class FeedForwardTest(CorrelationId: String, Weighted: Array[Float], InternalSubLayer:Int, FromInternalSubLayer:Int, Layer:Int, Shards: Int) extends ActivationCommand
  final case class getStats(replyTo: String, actorIndex : Int) extends ActivationCommand
  final case class getGradientsNeighbor(correlationId: String, gradients: Array[Float], layer:Int, internalSubLayerCaller:Int, localIndex:Int) extends ActivationCommand
  final case class fromGradientsNeighbor(correlationId: String, gradients: Array[Float], layer:Int,internalSubLayerCaller:Int, fromInternalSubLayer:Int) extends ActivationCommand


  def apply(actorId: String): Behavior[ActivationCommand] =
    Behaviors.setup { context =>
      context.system.receptionist ! Receptionist.Register(
        ServiceKey[ComputeActivation.ActivationCommand](actorId), context.self
      )
      new Activation(context)
    }
}

