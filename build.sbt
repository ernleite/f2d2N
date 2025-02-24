import com.typesafe.sbt.SbtMultiJvm.multiJvmSettings
import com.typesafe.sbt.SbtMultiJvm.MultiJvmKeys.MultiJvm

val akkaVersion = "2.10.0"

lazy val `akka-sample-cluster-scala` = project
  .in(file("."))
  .settings(multiJvmSettings: _*)
  .settings(
    resolvers += "Akka library repository".at("https://repo.akka.io/maven"),
    organization := "DeepGenAI LLC",
    scalaVersion := "2.13.12",
    Compile / scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Xlog-reflective-calls", "-Xlint"),
    Compile / javacOptions ++= Seq("-Xlint:unchecked", "-Xlint:deprecation"),
    run / javaOptions ++= Seq("-Xms16G", "-Xmx16G", "-XX:+UseG1GC","-Djava.library.path=./target/native"),
    libraryDependencies ++= Seq(
      "com.typesafe.akka" %% "akka-actor-typed"           % akkaVersion,
      "com.typesafe.akka" %% "akka-actor" % akkaVersion,
      "com.typesafe.akka" %% "akka-cluster-typed"         % akkaVersion,
      "com.typesafe.akka" %% "akka-serialization-jackson" % akkaVersion,
      "ch.qos.logback"    %  "logback-classic"             % "1.4.7",
      "org.scalatest"     %% "scalatest"                  % "3.2.15"     % Test,
      "com.typesafe.akka" %% "akka-stream" % akkaVersion,
      "org.scalanlp" %% "breeze" % "2.1.0",
      "com.amazonaws" % "aws-java-sdk-s3" % "1.12.761",
    //  "dev.ludovic.netlib" % "blas" % "3.0.3",
    //  "dev.ludovic.netlib" % "lapack" % "3.0.3",
    // "dev.ludovic.netlib" % "arpack" % "3.0.3",
      "org.slf4j" % "slf4j-api" % "2.0.13",
      "ai.djl" % "api" % "0.29.0",
      "ai.djl.pytorch" % "pytorch-engine" % "0.29.0"  ,
      "commons-cli"%"commons-cli" %"1.5.0",
      //"org.bytedeco" % "openblas" % "0.3.26-1.5.10",
      //"org.apache.commons" % "commons-compress" % "1.27.1",
     // "ai.djl.pytorch" % "pytorch-jni" % "2.3.1-0.29.0",
     // "ai.djl.pytorch" % "pytorch-native-cu124" % "2.4.0",
      // Use appropriate CUDA version
    //  "ai.djl.mxnet" % "mxnet-engine" % "0.29.0"  ,
    ),
    run / fork := true,
    Global / cancelable := false,
    parallelExecution in Global := true,
    // disable parallel tests
    Test / parallelExecution := false,
    licenses := Seq(("CC0", url("http://creativecommons.org/publicdomain/zero/1.0")))

  )
  .configs (MultiJvm)

