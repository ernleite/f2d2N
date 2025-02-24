package com.deeplearning.samples

import com.amazonaws.auth.{AWSStaticCredentialsProvider, BasicAWSCredentials}
import com.amazonaws.services.s3.AmazonS3ClientBuilder
import com.amazonaws.services.s3.model.{CSVInput, CSVOutput, InputSerialization, OutputSerialization, SelectObjectContentRequest}

import java.io.{BufferedReader, InputStreamReader}
import java.time.Instant
import scala.io.{Source}

class MnistData extends TrainingDataSet {

  this.trainingSize = 60000
  this.testSize = 10000
  this.SourceTrain = "mnist_train.csv"
  this.SourceTest = "mnist_test.csv"
  this.Input2D = "28,28"
  this.Input2DRows = 28
  this.Input2DCols = 28
  this.Channel = 1
  this.DatasetName = "MNIST"
  this.Size = this.Input2DRows*this.Input2DCols
  def loadTrainDataset(loadMode:String): Unit = {
    if (loadMode == "local") {
      val arr = Source.fromResource(SourceTrain).getLines().drop(1).toArray
      this.trainingSet = arr.map(_.split(",").map(_.toFloat))
    }
    else {
      val creds = new BasicAWSCredentials("AKIAWB5SSIYN7RFELY55", "HfduQ0mupRvI9Optkf5Zwf7AVUMJFS0I6vnp+BZm")
      val s3Client = AmazonS3ClientBuilder
        .standard()
        .withRegion("us-east-1")
        .withCredentials(new AWSStaticCredentialsProvider(creds))
        .build();

      // loading sharded data assigned to the input actor
      val contentRequest = new SelectObjectContentRequest()
      contentRequest.setBucketName("distributedweightednn")
      contentRequest.setKey(this.SourceTrain)
      contentRequest.setExpressionType("SQL")
      contentRequest.setExpression(s"SELECT * FROM s3object")
      val inputSer = new InputSerialization()
      val csvInput = new CSVInput()
      csvInput.setFileHeaderInfo("USE")
      csvInput.setFieldDelimiter(",")
      csvInput.setRecordDelimiter("\n")
      inputSer.setCsv(csvInput)
      contentRequest.setInputSerialization(inputSer)
      val outputSer = new OutputSerialization()
      val csvOutput = new CSVOutput()
      outputSer.setCsv(csvOutput)
      contentRequest.setOutputSerialization(outputSer)
      val result = s3Client.selectObjectContent(contentRequest)
      val startedAt = Instant.now
      val resultInputStream = result.getPayload.getRecordsInputStream
      val streamReader = new BufferedReader(new InputStreamReader(resultInputStream, "UTF-8"))
      val arr = streamReader.lines().skip(0).toArray
      this.trainingSet = arr.map(row => row.asInstanceOf[String].split(",").map(_.toFloat))
      val endedAt = Instant.now
      //context.log.info(s"Amazon S3 : Data sharded content $startIndex $endIndex loaded in " + ((Duration.between(startedAt, endedAt).toMillis) + " milliseconds"))
      streamReader.close()
      resultInputStream.close()
    }
    this.trainingSetLoaded = true
  }
  def loadTestDataset(loadMode:String): Unit = {
    if (loadMode == "local") {
      val arr = Source.fromResource(SourceTest).getLines().drop(1).toArray
      this.testSet = arr.map(_.split(",").map(_.toFloat))
    }
    else {
      val creds = new BasicAWSCredentials("AKIAWB5SSIYN7RFELY55", "HfduQ0mupRvI9Optkf5Zwf7AVUMJFS0I6vnp+BZm");
      val s3Client = AmazonS3ClientBuilder
        .standard()
        .withRegion("us-east-1")
        .withCredentials(new AWSStaticCredentialsProvider(creds))
        .build();

      // loading sharded data assigned to the input actor
      val contentRequest = new SelectObjectContentRequest()
      contentRequest.setBucketName("distributedweightednn")
      contentRequest.setKey(this.SourceTest)
      contentRequest.setExpressionType("SQL")
      contentRequest.setExpression(s"SELECT * FROM s3object")
      val inputSer = new InputSerialization()
      val csvInput = new CSVInput()
      csvInput.setFileHeaderInfo("USE")
      csvInput.setFieldDelimiter(",")
      csvInput.setRecordDelimiter("\n")
      inputSer.setCsv(csvInput)
      contentRequest.setInputSerialization(inputSer)
      val outputSer = new OutputSerialization()
      val csvOutput = new CSVOutput()
      outputSer.setCsv(csvOutput)
      contentRequest.setOutputSerialization(outputSer)
      val result = s3Client.selectObjectContent(contentRequest)
      val startedAt = Instant.now
      val resultInputStream = result.getPayload.getRecordsInputStream
      val streamReader = new BufferedReader(new InputStreamReader(resultInputStream, "UTF-8"))
      val endedAt = Instant.now
      //context.log.info(s"Amazon S3 : Data sharded test content $startIndex $endIndex loaded in " + ((Duration.between(startedAt, endedAt).toMillis) + " milliseconds"))
      val arr = streamReader.lines().skip(0).toArray
      this.testSet = arr.map(row => row.asInstanceOf[String].split(",").map(_.toFloat))
      streamReader.close()
      resultInputStream.close()
    }
    this.testSetLoaded = true
  }
  def getTrainingInput(index: Int): Array[Float] = {
    this.trainingSet(index)
  }
  def getTestInput(index:Int):Array[Float] = {
    this.testSet(index)
  }

  def getTrainingLabelData(col: Int): Array[Int] = {
    this.trainingSet.map(row => row(1).asInstanceOf[Int])
  }

  def getTestLabelData(col: Int): Array[Int] = {
    this.testSet.map(row => row(1).asInstanceOf[Int])
  }

}
