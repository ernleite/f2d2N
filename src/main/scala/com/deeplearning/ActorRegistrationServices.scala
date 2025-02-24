package com.deeplearning

import akka.actor.typed.{ActorRef,  Behavior}
import akka.actor.typed.receptionist.{Receptionist, ServiceKey}
import akka.actor.typed.scaladsl.{ActorContext, Behaviors}

object ComputeInputWeightedRegistryManager
{
  sealed trait MessageToLayer
  final case object FindActor extends MessageToLayer
  private case class ListingResponse(listing: Receptionist.Listing)
    extends MessageToLayer

  def apply(actorId: String): Behavior[ComputeInputWeightedRegistryManager.MessageToLayer] = Behaviors.setup {
    context: ActorContext[ComputeInputWeightedRegistryManager.MessageToLayer] =>

      // (1) we can’t initially get a reference to the Mouth actor, so
      // declare this variable as a var field, and using Option/None
      var layer: Option[ActorRef[ComputeInputs.InputCommand]] = None

      // (2) create an ActorRef that can be thought of as a Receptionist
      // Listing “adapter.” this will be used in the next line of code.
      // the Brain.ListingResponse(listing) part of the code tells the
      // Receptionist how to get back in touch with us after we contact
      // it in Step 4 below.
      // also, this line of code is long, so i wrapped it onto two lines.
      val listingAdapter: ActorRef[Receptionist.Listing] =
      context.messageAdapter { listing =>
        ComputeInputWeightedRegistryManager.ListingResponse(listing)
      }

      // (3) send a message to the Receptionist saying that we want
      // to subscribe to events related to Mouth.MouthKey, which
      // represents the Mouth actor.
      context.system.receptionist !
        Receptionist.Subscribe(ServiceKey[ComputeInputs.InputCommand](actorId), listingAdapter)

      Behaviors.receiveMessage { message =>
        val serviceKey = ServiceKey[ComputeInputs.InputCommand](actorId)

        message match {
          case FindActor =>
            // (4) send a Find message to the Receptionist, saying
            // that we want to find any/all listings related to
            // Mouth.MouthKey, i.e., the Mouth actor.
            //context.log.info(s"HiddenLayer: got a message")
            context.system.receptionist !
              Receptionist.Find(ServiceKey[ComputeInputs.InputCommand](actorId), listingAdapter)
            Behaviors.same
          case ListingResponse(serviceKey.Listing(listings)) =>

            // (5) after Step 4, the Receptionist sends us this
            // ListingResponse message. the `listings` variable is
            // a Set of ActorRef of type Mouth.MessageToMouth, which
            // you can interpret as “a set of Mouth ActorRefs.” for
            // this example i know that there will be at most one
            // Mouth actor, but in other cases there may be more
            // than one actor in this set.

            // i add this line just to be clear about `listings` type
            val xs: Set[ActorRef[ComputeInputs.InputCommand]] = listings
            // loop through all of the ActorRefs

            for (x <- xs) {
              // there should be only one ActorRef, so i assign it
              // to the `mouth` variable i created earlier
              layer = Some(x)
              context.log.info(s"Registering $actorId with path ${layer.head}")
              if (! Network.Layers.equals(actorId)) {
                Network.Layers += (actorId -> layer.head.path.name)
              }
              Network.LayersInputRef(actorId) = layer.head
            }
            Behaviors.same
        }
      }
  }


}

object ComputeWeightedRegistryManager
{
  sealed trait MessageToLayer
  final case object FindActor extends MessageToLayer
  private case class ListingResponse(listing: Receptionist.Listing)
    extends MessageToLayer

  def apply(actorId: String): Behavior[ComputeWeightedRegistryManager.MessageToLayer] = Behaviors.setup {
    context: ActorContext[ComputeWeightedRegistryManager.MessageToLayer] =>

      // (1) we can’t initially get a reference to the Mouth actor, so
      // declare this variable as a var field, and using Option/None
      var layer: Option[ActorRef[ComputeWeighted.WeightedCommand]] = None

      // (2) create an ActorRef that can be thought of as a Receptionist
      // Listing “adapter.” this will be used in the next line of code.
      // the Brain.ListingResponse(listing) part of the code tells the
      // Receptionist how to get back in touch with us after we contact
      // it in Step 4 below.
      // also, this line of code is long, so i wrapped it onto two lines.
      val listingAdapter: ActorRef[Receptionist.Listing] =
      context.messageAdapter { listing =>
        ComputeWeightedRegistryManager.ListingResponse(listing)
      }

      // (3) send a message to the Receptionist saying that we want
      // to subscribe to events related to Mouth.MouthKey, which
      // represents the Mouth actor.
      context.system.receptionist !
        Receptionist.Subscribe(ServiceKey[ComputeWeighted.WeightedCommand](actorId), listingAdapter)

      Behaviors.receiveMessage { message =>
        val serviceKey = ServiceKey[ComputeWeighted.WeightedCommand](actorId)

        message match {
          case FindActor =>
            // (4) send a Find message to the Receptionist, saying
            // that we want to find any/all listings related to
            // Mouth.MouthKey, i.e., the Mouth actor.
            context.system.receptionist !
              Receptionist.Find(ServiceKey[ComputeWeighted.WeightedCommand](actorId), listingAdapter)
            Behaviors.same
          case ListingResponse(serviceKey.Listing(listings)) =>

            // (5) after Step 4, the Receptionist sends us this
            // ListingResponse message. the `listings` variable is
            // a Set of ActorRef of type Mouth.MessageToMouth, which
            // you can interpret as “a set of Mouth ActorRefs.” for
            // this example i know that there will be at most one
            // Mouth actor, but in other cases there may be more
            // than one actor in this set.

            // i add this line just to be clear about `listings` type
            val xs: Set[ActorRef[ComputeWeighted.WeightedCommand]] = listings
            // loop through all of the ActorRefs

            for (x <- xs) {
              // there should be only one ActorRef, so i assign it
              // to the `mouth` variable i created earlier
              layer = Some(x)
              context.log.info(s"Registering $actorId with path ${layer.head}")
              Network.LayersIntermediateRef(actorId) = layer.head
            }
            Behaviors.same
        }
      }
  }


}

object ComputeActivationRegistryManager
{
  sealed trait MessageToLayer
  final case object FindActor extends MessageToLayer
  private case class ListingResponse(listing: Receptionist.Listing)
    extends MessageToLayer

  def apply(actorId: String): Behavior[ComputeActivationRegistryManager.MessageToLayer] = Behaviors.setup {
    context: ActorContext[ComputeActivationRegistryManager.MessageToLayer] =>

      // (1) we can’t initially get a reference to the Mouth actor, so
      // declare this variable as a var field, and using Option/None
      var layer: Option[ActorRef[ComputeActivation.ActivationCommand]] = None

      // (2) create an ActorRef that can be thought of as a Receptionist
      // Listing “adapter.” this will be used in the next line of code.
      // the Brain.ListingResponse(listing) part of the code tells the
      // Receptionist how to get back in touch with us after we contact
      // it in Step 4 below.
      // also, this line of code is long, so i wrapped it onto two lines.
      val listingAdapter: ActorRef[Receptionist.Listing] =
      context.messageAdapter { listing =>
        ComputeActivationRegistryManager.ListingResponse(listing)
      }

      // (3) send a message to the Receptionist saying that we want
      // to subscribe to events related to Mouth.MouthKey, which
      // represents the Mouth actor.
      context.system.receptionist !
        Receptionist.Subscribe(ServiceKey[ComputeActivation.ActivationCommand](actorId), listingAdapter)

      Behaviors.receiveMessage { message =>
        val serviceKey = ServiceKey[ComputeActivation.ActivationCommand](actorId)

        message match {
          case FindActor =>
            // (4) send a Find message to the Receptionist, saying
            // that we want to find any/all listings related to
            // Mouth.MouthKey, i.e., the Mouth actor.
            context.system.receptionist !
              Receptionist.Find(ServiceKey[ComputeActivation.ActivationCommand](actorId), listingAdapter)
            Behaviors.same
          case ListingResponse(serviceKey.Listing(listings)) =>

            // (5) after Step 4, the Receptionist sends us this
            // ListingResponse message. the `listings` variable is
            // a Set of ActorRef of type Mouth.MessageToMouth, which
            // you can interpret as “a set of Mouth ActorRefs.” for
            // this example i know that there will be at most one
            // Mouth actor, but in other cases there may be more
            // than one actor in this set.

            // i add this line just to be clear about `listings` type
            val xs: Set[ActorRef[ComputeActivation.ActivationCommand]] = listings
            // loop through all of the ActorRefs

            for (x <- xs) {
              // there should be only one ActorRef, so i assign it
              // to the `mouth` variable i created earlier
              layer = Some(x)
              context.log.info(s"Registering $actorId with path ${layer.head}")
              Network.LayersHiddenRef(actorId) = layer.head
            }
            Behaviors.same
        }
      }
  }
}

object ComputeOutputRegistryManager
{
  sealed trait MessageToLayer
  final case object FindActor extends MessageToLayer
  private case class ListingResponse(listing: Receptionist.Listing)
    extends MessageToLayer

  def apply(actorId: String): Behavior[ComputeOutputRegistryManager.MessageToLayer] = Behaviors.setup {
    context: ActorContext[ComputeOutputRegistryManager.MessageToLayer] =>

      // (1) we can’t initially get a reference to the Mouth actor, so
      // declare this variable as a var field, and using Option/None
      var layer: Option[ActorRef[ComputeOutput.OutputCommand]] = None

      // (2) create an ActorRef that can be thought of as a Receptionist
      // Listing “adapter.” this will be used in the next line of code.
      // the Brain.ListingResponse(listing) part of the code tells the
      // Receptionist how to get back in touch with us after we contact
      // it in Step 4 below.
      // also, this line of code is long, so i wrapped it onto two lines.
      val listingAdapter: ActorRef[Receptionist.Listing] =
      context.messageAdapter { listing =>
        ComputeOutputRegistryManager.ListingResponse(listing)
      }

      // (3) send a message to the Receptionist saying that we want
      // to subscribe to events related to Mouth.MouthKey, which
      // represents the Mouth actor.
      context.system.receptionist !
        Receptionist.Subscribe(ServiceKey[ComputeOutput.OutputCommand](actorId), listingAdapter)

      Behaviors.receiveMessage { message =>
        val serviceKey = ServiceKey[ComputeOutput.OutputCommand](actorId)

        message match {
          case FindActor =>
            // (4) send a Find message to the Receptionist, saying
            // that we want to find any/all listings related to
            // Mouth.MouthKey, i.e., the Mouth actor.
            context.system.receptionist !
              Receptionist.Find(ServiceKey[ComputeOutput.OutputCommand](actorId), listingAdapter)
            Behaviors.same
          case ListingResponse(serviceKey.Listing(listings)) =>

            // (5) after Step 4, the Receptionist sends us this
            // ListingResponse message. the `listings` variable is
            // a Set of ActorRef of type Mouth.MessageToMouth, which
            // you can interpret as “a set of Mouth ActorRefs.” for
            // this example i know that there will be at most one
            // Mouth actor, but in other cases there may be more
            // than one actor in this set.

            // i add this line just to be clear about `listings` type
            val xs: Set[ActorRef[ComputeOutput.OutputCommand]] = listings
            // loop through all of the ActorRefs

            for (x <- xs) {
              // there should be only one ActorRef, so i assign it
              // to the `mouth` variable i created earlier
              layer = Some(x)
              context.log.info(s"Registering $actorId with path ${layer.head}")
              Network.LayersOutputRef(actorId) = layer.head
            }
            Behaviors.same
        }
      }
  }
}

object EpochsRegistryManager
{
  sealed trait MessageToLayer
  final case object FindActor extends MessageToLayer
  private case class ListingResponse(listing: Receptionist.Listing)
    extends MessageToLayer

  def apply(actorId: String): Behavior[EpochsRegistryManager.MessageToLayer] = Behaviors.setup {
    context: ActorContext[EpochsRegistryManager.MessageToLayer] =>

      // (1) we can’t initially get a reference to the Mouth actor, so
      // declare this variable as a var field, and using Option/None
      var layer: Option[ActorRef[ComputeEpochs.TrainCommand]] = None

      // (2) create an ActorRef that can be thought of as a Receptionist
      // Listing “adapter.” this will be used in the next line of code.
      // the Brain.ListingResponse(listing) part of the code tells the
      // Receptionist how to get back in touch with us after we contact
      // it in Step 4 below.
      // also, this line of code is long, so i wrapped it onto two lines.
      val listingAdapter: ActorRef[Receptionist.Listing] =
      context.messageAdapter { listing =>
        EpochsRegistryManager.ListingResponse(listing)
      }

      // (3) send a message to the Receptionist saying that we want
      // to subscribe to events related to Mouth.MouthKey, which
      // represents the Mouth actor.
      context.system.receptionist !
        Receptionist.Subscribe(ServiceKey[ComputeEpochs.TrainCommand](actorId), listingAdapter)

      Behaviors.receiveMessage { message =>
        val serviceKey = ServiceKey[ComputeEpochs.TrainCommand](actorId)

        message match {
          case FindActor =>
            // (4) send a Find message to the Receptionist, saying
            // that we want to find any/all listings related to
            // Mouth.MouthKey, i.e., the Mouth actor.
            context.system.receptionist !
              Receptionist.Find(ServiceKey[ComputeEpochs.TrainCommand](actorId), listingAdapter)
            Behaviors.same
          case ListingResponse(serviceKey.Listing(listings)) =>

            // (5) after Step 4, the Receptionist sends us this
            // ListingResponse message. the `listings` variable is
            // a Set of ActorRef of type Mouth.MessageToMouth, which
            // you can interpret as “a set of Mouth ActorRefs.” for
            // this example i know that there will be at most one
            // Mouth actor, but in other cases there may be more
            // than one actor in this set.

            // i add this line just to be clear about `listings` type
            val xs: Set[ActorRef[ComputeEpochs.TrainCommand]] = listings
            // loop through all of the ActorRefs

            for (x <- xs) {
              // there should be only one ActorRef, so i assign it
              // to the `mouth` variable i created earlier
              layer = Some(x)
              context.log.info(s"Registering $actorId with path ${layer.head}")
              Network.EpochsRef(actorId) = layer.head
            }
            Behaviors.same
        }
      }
  }
}
