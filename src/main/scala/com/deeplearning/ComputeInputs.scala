package com.deeplearning

import akka.actor.typed.Behavior
import akka.actor.typed.receptionist.{Receptionist, ServiceKey}
import akka.actor.typed.scaladsl.{AbstractBehavior, ActorContext, Behaviors}
import com.deeplearning.ComputeEpochs.SetStats
import com.deeplearning.layer.{InputLayer, LayerFactory}
import com.deeplearning.samples.TrainingDataSet

class Input(context: ActorContext[ComputeInputs.InputCommand]) extends AbstractBehavior[ComputeInputs.InputCommand](context) {
  import ComputeInputs._
  private var layer:InputLayer = _
  private val parameters = scala.collection.mutable.HashMap.empty[String, String]
  private var eventFF = 0
  private var eventBP = 0

  override def onMessage(msg: ComputeInputs.InputCommand): Behavior[ComputeInputs.InputCommand] = {
    msg match {
      case Weights(epoch:Int, correlationId: String, yLabel:Int, startIndex:Int, endIndex:Int, data: Array[Float], index:Int,layer:Int, internalSubLayer:Int) =>
        if (this.layer == null) {
          this.layer = LayerFactory.getInputLayer(Network.InputLayerType)
        }
        eventFF = eventFF + 1
        this.layer.computeInputWeights(epoch, correlationId, yLabel, startIndex, endIndex, data, index, layer, internalSubLayer, parameters)
        this

      case BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Float, internalSubLayer: Int, fromInternalSubLayer: Int, params : scala.collection.mutable.HashMap[String,String]) =>
        eventBP = eventBP + 1

        val completed = this.layer.BackPropagate(correlationId,delta,learningRate,regularisation,nInputs,internalSubLayer, fromInternalSubLayer, params)
        if (completed) {
          val epoch = Network.EpochsRef("epoch_0")
          epoch ! ComputeEpochs.NotifyMiniBatchCompleted(internalSubLayer, params)
        }
        this

      case FeedForwardTest(correlationId: String, startIndex: Int, endIndex: Int, index: Int, data: Array[Float], internalSubLayer: Int,  layer: Int) =>
        this.layer.FeedForwardTest(correlationId, startIndex, endIndex, index, data, internalSubLayer, layer)
        this

      case getStats(replyTo:String, actorIndex : Int) =>
        val epoch = Network.EpochsRef(replyTo)
        epoch ! SetStats(eventFF,eventBP, "inputLayer_"+ actorIndex)
        this
    }
  }

}

trait ComputeInputSerializable
object ComputeInputs {
  sealed trait InputCommand extends ComputeInputSerializable
  final case class Weights(Epoch:Int, CorrelationId: String, yLabel:Int, startIndex:Int, endIndex:Int, data: Array[Float], Index:Int, Layer:Int, InternalSubLayer:Int) extends InputCommand
  final case class BackPropagate(CorrelationId: String, delta: Array[Float],  learningRate : Float, regularisation:Float, nInputs:Float, InternalSubLayer:Int,  FromInternalSubLayer:Int, Params : scala.collection.mutable.HashMap[String,String]) extends InputCommand
  final case class FeedForwardTest(CorrelationId: String, startIndex:Int, endIndex:Int, Index:Int,  data: Array[Float],  InternalSubLayer:Int, Layer:Int) extends InputCommand
  final case class getStats(replyTo: String, actorIndex : Int) extends InputCommand

  def apply(actorId: String): Behavior[InputCommand] =
    Behaviors.setup { context =>
      context.system.receptionist ! Receptionist.Register(
        ServiceKey[ComputeInputs.InputCommand](actorId), context.self
      )
      new Input(context)
    }

}

