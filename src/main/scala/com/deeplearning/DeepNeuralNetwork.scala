package com.deeplearning

import akka.actor.typed.{Behavior}
import akka.actor.typed.scaladsl.{Behaviors}

import scala.io.Source

object DeepNeuralNetwork  {
  final case class Build(name: String)

  def apply(): Behavior[Build] = Behaviors.setup { context =>
    val layers = Source.fromResource("network.conf").getLines.toArray[String]

    // build the input layers actors
    for (x <- 0 until Network.InputLayerDim) {
      val name = "inputLayer_" + x

      val registering = context.spawn(ComputeInputWeightedRegistryManager(name), java.util.UUID.randomUUID.toString)
      if (layers contains name) {
        registering ! ComputeInputWeightedRegistryManager.FindActor
        context.log.info(s"Registering actor entry for $name")
        val instance = context.spawn(ComputeInputs(name), name)
        Network.Layers += (name -> instance.path.name)
      }
    }

    // build the intermediate layers actors
    var idxW = 2
    for (l <- 1 until ((Network.HiddenLayers.length)+1)) {
        for (x <- 0 until Network.getHiddenLayersDim(idxW, "hidden")) {
        val name = "weightedLayer_" + idxW + "_" + x
        val registering = context.spawn(ComputeWeightedRegistryManager(name), java.util.UUID.randomUUID.toString)
        if (layers contains name) {
          registering ! ComputeWeightedRegistryManager.FindActor
          context.log.info(s"Registering actor entry for $name")
          val instance = context.spawn(ComputeWeighted(name), name)
          Network.Layers += (name -> instance.path.name)
        }
      }
      idxW +=2
    }

    var idxA = 1
    // build the hidden layers actors
    for (l <- 1 until ((Network.HiddenLayers.length)+1)) {
      for (x <- 0 until 1) {
        val name = "hiddenLayer_" + idxA + "_" + x
        val registering = context.spawn(ComputeActivationRegistryManager(name), java.util.UUID.randomUUID.toString)
        if (layers contains  name) {
          registering ! ComputeActivationRegistryManager.FindActor
          context.log.info(s"Registering actor entry for $name")
          val instance = context.spawn(ComputeActivation(name), name)
          Network.Layers += (name-> instance.path.name)
        }
      }
      idxA +=2
    }

    // build the output layers
    for (x <- 0 until Network.OutputLayerDim) {
      val name = "outputLayer_" + x
      val registering = context.spawn(ComputeOutputRegistryManager(name), java.util.UUID.randomUUID.toString)
      if (layers contains name) {
        registering ! ComputeOutputRegistryManager.FindActor
        context.log.info(s"Registering actor entry for $name")
        val instance = context.spawn(ComputeOutput(name), java.util.UUID.randomUUID.toString)
        Network.Layers += ("outputLayer_" + x -> instance.path.name)
      }
    }

    Behaviors.receiveMessage {
      message =>
        println(message.name)

        val role = Source.fromResource("role.conf").getLines().take(1)
        val epochName = "epoch_0"
        val registering = context.spawn(EpochsRegistryManager(epochName), java.util.UUID.randomUUID.toString)
        registering ! EpochsRegistryManager.FindActor

        if (role.contains("master")) {
          val instance = context.spawn(ComputeEpochs(epochName), java.util.UUID.randomUUID.toString)
          Network.EpochsRef += (epochName -> instance)
          instance ! ComputeEpochs.Train("Building neural network : Done !")
        }
        Behaviors.same
    }
  }
}

