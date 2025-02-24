package com.deeplearning

import akka.actor.typed.Behavior
import akka.actor.typed.receptionist.{Receptionist, ServiceKey}
import akka.actor.typed.scaladsl.{AbstractBehavior, ActorContext, Behaviors}
import com.deeplearning.ComputeInputs.getStats
import com.deeplearning.ComputeOutput.{ComputeLoss, GetErrorRate}
import com.deeplearning.Network.limitedDataset
import com.deeplearning.samples.{CifarData, MnistData, TrainingDataSet}

import java.time.{Duration, Instant}


class Epochs(context: ActorContext[ComputeEpochs.TrainCommand]) extends AbstractBehavior[ComputeEpochs.TrainCommand](context) {
  import ComputeEpochs._

  private val yTrain =  scala.collection.mutable.HashMap.empty[String, Int]
  private val yTest =  scala.collection.mutable.HashMap.empty[String, Int]
  private val epochsStats =  scala.collection.mutable.HashMap.empty[Int,String]
  private var X = Array.ofDim[Float](Network.InputLayer)
  private var yTrainingSet = Array[Int]()
  private var yTestSet = Array[Int]()
  private var yTestIndexSet = Array[Int]()
  private var eInitialized:Boolean = false
  private var tInitialized:Boolean = false
  private var startIndex:Int = 0
  private var actorCount:Int = 0
  private var startedAt = Instant.now
  private var endedAt = Instant.now
  private var epochsStartedAt = Instant.now
  private var epochsEndedAt = Instant.now
  private var truePredict = 0
  private var testDoneCount = 0
  private var epochDoneCount = 0
  private var lastEpochIndex = 0
  private var inputsTrainId = Array[Int]()
  private var inputsTestId = Array[Int]()
  private var miniBatchCompleted = 0
  private var eventFF = 0
  private var eventBP = 0
  private var actorsCount = 0

  private var dataSet:TrainingDataSet = if (Network.trainingSample == "Mnist") {
    dataSet = new MnistData()
    dataSet
  } else  {
    dataSet = new CifarData()
    dataSet
  }

  override def onMessage(msg: ComputeEpochs.TrainCommand): Behavior[ComputeEpochs.TrainCommand] = {
    msg match {
      case Train(name:String) =>
        if (!eInitialized) {
          dataSet.loadTrainDataset(Network.InputLoadMode)
          yTrainingSet = dataSet.getTrainingLabelData(1)
          eInitialized=true
          epochsStartedAt = Instant.now()
          if (Network.GpuMode) context.log.info("************* GPU MODE ACTIVATED *************")
          else context.log.info("************* CPU MODE ACTIVATED *************")
        }

        //Only take a set of the training size / SGB
       // inputsTrainId = Network.generateArray(0,dataSet.trainingSize-1).take(Network.MiniBatchRange)
       // inputsTrainId = Array(1) ++ Array(2)++ Array(3)++ Array(4)++ Array(5)++ Array(6)++ Array(7)++ Array(8)++ Array(9)++ Array(10)
        inputsTrainId = if (Network.limitedDataset) Array(1)++ Array(2) else Network.generateArray(0,dataSet.trainingSize-1).take(Network.MiniBatchRange)

        if (Network.LearningRateDecay) {
          Network.LearningRate = Network.stepDecay(epochDoneCount,Network.InitialLearningRate,Network.drop,Network.epochs_drop)
          if (Network.debugDelay) {
            context.log.info("-----------------------------------------------------")
            context.log.info("stepDecay : New LearningRate = " + Network.LearningRate)
            context.log.info("-----------------------------------------------------")
          }
        }
        else {
          context.log.info("LearningRate = " + Network.LearningRate)
        }

        startedAt = Instant.now()
        yTrain.clear()
        startIndex = 0
        context.self! NextMiniBatch()
        this
      case Test(name: String) =>
        if (!tInitialized) {
          dataSet.loadTestDataset(Network.InputLoadMode)
          yTestSet = dataSet.getTestLabelData(1)
          yTest.clear()
          tInitialized = true
        }
        truePredict = 0
        testDoneCount = 0
        inputsTestId = Network.generateArray(0,dataSet.testSize-1)
        startIndex = 0
        context.self ! NextMiniBatchTest()
        this

      case NextMiniBatch() =>
        val range = LayerManager.GetInputLayerStep()
        //calculate first sub hidden layers
        val intermediatelayers = Network.InputLayerDim
        //println(startIndex)

        if (startIndex==0) {
          context.log.info("Training starting of " + Network.MiniBatchRange)
        }

        if (epochDoneCount == 0) {
          FileHelper.appendTrainingHeader()
          FileHelper.appendTestHeader()
        }

        if (epochDoneCount>0 && lastEpochIndex < epochDoneCount) {
          lastEpochIndex = epochDoneCount
          endedAt = Instant.now
          val duration = Duration.between(startedAt, endedAt).toSeconds
          epochsStats(epochDoneCount) +=  "," + duration
          context.log.info("---------------------------------------")
          context.log.info(s"Epoch " + epochDoneCount + "/" + Network.Epochs  +" - Training completed on " + Network.MiniBatchRange + " training samples in " + duration + " sec." )
          context.log.info("---------------------------------------")
          FileHelper.appendToTrainingCsv(epochsStats(epochDoneCount))

          if (epochDoneCount == Network.Epochs) {
            epochsEndedAt = Instant.now()
            startIndex = 0
            context.log.info("---------------------------------------")
            val duration = Duration.between(epochsStartedAt, epochsEndedAt).toMinutes
            context.log.info(s"Total Training phase completed on $duration mins.")
            context.log.info("---------------------------------------")
            context.log.info("")
            context.log.info("---------------------------------------")
            context.log.info("Launching Testing phase on UNSEAN data")
            context.log.info("---------------------------------------")
            context.self! Test("Test DNN")
          }
          else
            context.self! Train("Train DNN")
        }
        else {

          var mainOffset = 0
          var index = startIndex
          var endIndex = Network.MiniBatch

          if (index + endIndex >= Network.MiniBatchRange ) {
            endIndex = Network.MiniBatchRange - index
          }

          for (i <- 0 until endIndex) {
            val correlationId = java.util.UUID.randomUUID.toString
            var offset = 0
            var offsetRange = 0
            for (x: Int <- 0 until intermediatelayers) {
              offsetRange = offsetRange + range
              val name = "inputLayer_" + x
              val computeWeighted = Network.LayersInputRef(name)
              yTrain += (correlationId -> yTrainingSet(inputsTrainId(index)))
              val inputs = dataSet.getTrainingInput(inputsTrainId(index))
              val input = inputs.slice(offset + 2, offsetRange + 2) //normal
              computeWeighted ! ComputeInputs.Weights(epochDoneCount, correlationId, yTrain(correlationId), offset, offsetRange, input, inputsTrainId(index), 0, x)

              offset += range
              mainOffset += offset
              //context.log.info(i.toString + " " + index.toString + " " + intermediatelayers + " " + endIndex)
            }
            index += 1
          }

          if (startIndex%Network.minibatchBuffer == 0)
            context.log.info(s"Minibach completed. From $startIndex to " + (startIndex + Network.minibatchBuffer) + ". Remaining : " + (Network.MiniBatchRange - (startIndex + Network.minibatchBuffer)))
          }

        startIndex += Network.MiniBatch
        this

      case NextMiniBatchTest() =>
        val range = LayerManager.GetInputLayerStep()
        //calculate first sub hidden layers
        val intermediatelayers = Network.InputLayerDim

        var mainOffset = 0
        var index = startIndex
        var endIndex = Network.MiniBatch

        if (index + endIndex >= dataSet.testSize) {
          endIndex = dataSet.testSize - index
        }

        for (i <- yTestSet.indices) {
          val correlationId = java.util.UUID.randomUUID.toString
          var offset = 0
          var offsetRange = 0
          for (x: Int <- 0 until intermediatelayers) {
            offsetRange = offsetRange + range
            val name = "inputLayer_" + x
            val computeWeighted = Network.LayersInputRef(name)
            yTest += (correlationId -> yTestSet(inputsTestId(index)))
            val inputs = dataSet.getTestInput(inputsTestId(index))
            val input = inputs.slice(offset + 2, offsetRange + 2) //normal
            computeWeighted ! ComputeInputs.FeedForwardTest(correlationId, offset, offsetRange, inputsTestId(index),input, x,0)

            offset += range
            mainOffset += offset
          }
          index += 1
        }
        startIndex += Network.MiniBatch
        this

      case NotifyFeedForward(correlationId: String, params:scala.collection.mutable.HashMap[String,String], replyTo:String,internalSubLayer:Int) =>
        var arr = Array[Float]()
        arr = Array.fill[Float](Network.OutputLayer)(0)
        arr(yTrain(correlationId)) = 1
        // context.log.info(s"Receiving label request from $correlationId for index " + y(correlationId))
        val actor = Network.LayersOutputRef(replyTo)
        actor ! ComputeLoss(correlationId, arr, Network.MiniBatchRange, Network.LearningRate, Network.Regularisation, internalSubLayer, params)
        this

      case NotifyFeedForwardTest(correlationId: String, labelFound: Float, replyTo: String) =>
        var arr = Array[Float]()
        arr = Array.fill[Float](Network.OutputLayer)(0)
        val labelSearched = yTest(correlationId)
        var goodMatch = false
        if (testDoneCount == 0) {
          //epochsStats.clear()
          FileHelper.appendTestHeader()
        }

        testDoneCount += 1
        if (labelFound == labelSearched) {
          truePredict += 1
          goodMatch = true
        }

        if (goodMatch) {
            //println(testDoneCount +  " Search Label : " + yTest(correlationId)  + " Found Label : " + labelFound + " " + goodMatch)
        }

        if (testDoneCount == dataSet.testSize) {
          context.log.info("---------------------------------------")
          context.log.info(s"Test completed on ${dataSet.testSize} training samples")
          context.log.info(s"Accuracy : $truePredict correct ${dataSet.testSize-truePredict} incorrects. Ratio : " + ((truePredict.toFloat / dataSet.testSize.toFloat) * 100) + " %")
          context.log.info("---------------------------------------")

          val accuracy = truePredict.toFloat / dataSet.testSize.toFloat * 100
          FileHelper.appendToTestCsv( truePredict.toString + "," + (dataSet.testSize - truePredict).toString + "," + accuracy.toString )

          testDoneCount = 0
          truePredict = 0
          startIndex = 0
          //yTest.clear()
        }
        else
          NextMiniBatchTest()

        this

      case NotifyMiniBatchCompleted(inputActor:Int,params : scala.collection.mutable.HashMap[String,String]) =>
        actorCount +=1

        if (actorCount == Network.InputLayerDim) {
          miniBatchCompleted = miniBatchCompleted + 1
          if (Network.MiniBatchRange/Network.MiniBatch == miniBatchCompleted) {
            val output = Network.LayersOutputRef("outputLayer_0")
            output! GetErrorRate(context.self, params)
          }
          else {
            //context.log.info(actorCount.toString)
            context.self ! NextMiniBatch()
          }
          actorCount = 0
        }
        this

      case ErrorRate(errorRate: Float,params:scala.collection.mutable.HashMap[String,String]) =>
        context.log.info("")
        context.log.info("********** Error Rate : " + errorRate + " **********")
        context.log.info("")
        epochDoneCount = epochDoneCount + 1
        epochsStats += (epochDoneCount) -> ((epochDoneCount).toString + "," + errorRate.toString)
        miniBatchCompleted = 0
        yTrain.clear()

        context.log.info("********** Events ********** ")
        this.eventFF = 0
        this.eventBP = 0
        actorsCount = 0
        actorCount = 0
        context.self! GetStats()
        this

      case GetStats() =>
        for (x <- 0 until Network.InputLayerDim) {
          actorsCount+=1
          val name = "inputLayer_" + x
          val actor = Network.LayersInputRef(name)
          actor ! ComputeInputs.getStats("epoch_0", x)
        }
        var idxW = 2
        for (l <- 2 until ((Network.HiddenLayers.length)+1) by 2) {
          val test = Network.getHiddenLayersDim(l, "weighted")
          for (x <- 0 until Network.getHiddenLayersDim(l, "hidden")) {
            actorsCount+=1
            val name = "weightedLayer_" + idxW + "_" + x
            val actor = Network.LayersIntermediateRef(name)
            actor ! ComputeWeighted.getStats("epoch_0",x)
          }
          idxW +=2
        }

        var idxA = 1
        // build the hidden layers actors
        for (l <- 1 until ((Network.HiddenLayers.length)+1)) {
          for (x <- 0 until 1) {
            actorsCount+=1
            val name = "hiddenLayer_" + idxA + "_" + x
            val actor = Network.LayersHiddenRef(name)
            actor ! ComputeActivation.getStats("epoch_0",x)
          }
          idxA +=2
        }
        for (x <- 0 until Network.OutputLayerDim) {
          actorsCount+=1
          val name = "outputLayer_" + x
          val actor2 = Network.LayersOutputRef(name)
          actor2 ! ComputeOutput.getStats("epoch_0",x)
        }
        this

    case SetStats(eventFF:Int, eventBP:Int, fromActor : String) =>
        actorCount+=1
        this.eventFF += eventFF
        this.eventBP += eventBP

        if (actorsCount == actorCount) {
          actorCount = 0
          context.log.info(this.eventFF + " " +this.eventBP)
          if (epochDoneCount == lastEpochIndex) {
            context.log.info("----------------- Statistics ----------------------")
            context.log.info("Feed-Forward events total : " + this.eventFF)
            context.log.info("Back-Propagation events total : " + this.eventBP)
          }
          epochsStats(epochDoneCount) +=  "," + this.eventFF + "," + this.eventBP
          context.self ! NextMiniBatch()
        }
        this
      }


    }
  }

trait EpochsSerializable
object ComputeEpochs {
  sealed trait TrainCommand extends EpochsSerializable
  final case class Train(name: String) extends TrainCommand
  final case class Test(name: String) extends TrainCommand
  final case class NotifyFeedForward(name: String, params:scala.collection.mutable.HashMap[String,String], replyTo:String,InternalSubLayer:Int) extends TrainCommand
  final case class NotifyFeedForwardTest(name: String, labelFound: Float, replyTo:String) extends TrainCommand
  final case class NextMiniBatch() extends TrainCommand
  final case class NextMiniBatchTest() extends TrainCommand
  final case class GetStats() extends TrainCommand
  final case class SetStats(eventFF:Int, eventBP:Int, fromActor: String) extends TrainCommand
  final case class ErrorRate(errorRate:Float,params:scala.collection.mutable.HashMap[String,String]) extends TrainCommand
  final case class NotifyMiniBatchCompleted(InputActor:Int,params : scala.collection.mutable.HashMap[String,String]) extends TrainCommand
  def apply(actorId: String): Behavior[ComputeEpochs.TrainCommand] =
    Behaviors.setup { context =>
      context.system.receptionist !  Receptionist.Register(
        ServiceKey[ComputeEpochs.TrainCommand](actorId), context.self
      )
      new Epochs(context)
    }
  }
