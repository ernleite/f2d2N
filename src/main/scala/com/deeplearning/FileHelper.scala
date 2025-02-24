package com.deeplearning
import java.io.{FileWriter}
import scala.io.Source
object FileHelper {
  def appendTrainingHeader(): Unit = {
    val file = new FileWriter("src/main/resources/training.txt")
    file.write("epoch,error,eventFF,eventBP,duration\r\n")
    file.close()
  }
  def appendToTrainingCsv(data: String): Unit = {
    val file = new FileWriter("src/main/resources/training.txt", true)
    file.write(data + "\r\n")
    file.close()
  }

  def appendTestHeader(): Unit = {
    val file = new FileWriter("src/main/resources/test.txt")
    file.write("correct,incorrect,accuracy\r\n")
    file.close()
  }

  def appendToTestCsv(data: String): Unit = {
    val file = new FileWriter("src/main/resources/test.txt", true)
    file.write(data + "\r\n")
    file.close()
  }
}
