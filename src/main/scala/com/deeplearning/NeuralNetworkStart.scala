package com.deeplearning

import akka.actor.typed.ActorSystem
import com.deeplearning.DeepNeuralNetwork.Build

object NeuralNetworkStart extends App {
  val dnn : ActorSystem[DeepNeuralNetwork.Build] = ActorSystem(DeepNeuralNetwork(), "ClusterSystem")
  dnn ! Build("Construct Deep Leaning Graph : OK")
  println(s"First: $dnn")
}