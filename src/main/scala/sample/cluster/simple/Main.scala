package sample.cluster.simple

import akka.actor.typed.scaladsl.Behaviors
import akka.actor.typed.{ActorSystem, Behavior}
import com.deeplearning.{DeepNeuralNetwork, Network}
import com.deeplearning.DeepNeuralNetwork.Build
import com.typesafe.config.ConfigFactory

object StartMain {
  object RootBehavior {
    def apply(): Behavior[Nothing] = Behaviors.setup[Nothing] { context =>
      // Create an actor that handles cluster domain events
      context.spawn(ClusterListener(), "ClusterListener")

      while (!ClusterListener.DNNStarted) {
        if (ClusterListener.Nodes == Network.clusterNodesDim) {
            Thread.sleep(2000)
            val dnn = context.spawn(DeepNeuralNetwork(), "ClusterSystem")
            dnn ! Build("Construct Deep Leaning Graph : OK")
            ClusterListener.DNNStarted = true
        }
        Thread.sleep(1000)
      }
      Behaviors.empty
    }
  }

  def main(args: Array[String]): Unit = {
    val ports =
      if (args.isEmpty)
        Seq(25252)
      else
        args.toSeq.map(_.toInt)
    ports.foreach(startup)
  }

  def startup(port: Int): Unit = {
    // Override the configuration of the port
    val config = ConfigFactory.parseString(s"""
      akka.remote.artery.canonical.port=$port
      """).withFallback(ConfigFactory.load())

    // Create an Akka system
    ActorSystem[Nothing](RootBehavior(), "DeepNeuralNetwork", config)
  }

}
