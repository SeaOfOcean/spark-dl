/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.pvanet

object TrainAltOpt {

  def trainRpn(imdbName: String, initModel: String, maxIters: Int): String = {
    ""
  }

  def rpnGenerate(imdbName: String, rpnModelPath: String): String = {
    ""
  }

  def trainFastRcnn(
    imdbName: String,
    initModel: String,
    maxIters: Int,
    rpnProposals: String): String = {
    ""
  }


  def main(args: Array[String]): Unit = {
    val imdbName = "voc_2007"
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 RPN, init from ImageNet model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpnStage1Model = trainRpn(imdbName, "pretrained_path", 10)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 RPN, generate proposals")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpnStage1Proposals = rpnGenerate(imdbName, rpnStage1Model)
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    val fastRcnnStage1Model = trainFastRcnn(
      imdbName, "pretrained_path", 10, rpnStage1Proposals)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 RPN, init from stage 1 Fast R-CNN model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    val rpnStage2Model = trainRpn(imdbName, fastRcnnStage1Model, 10)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 RPN, generate proposals")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpnStage2Proposals = rpnGenerate(imdbName, rpnStage2Model)



    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val fastRcnnStage2Model = trainFastRcnn(
      imdbName, rpnStage2Model, 10, rpnStage2Proposals)
    val finalModelPath = fastRcnnStage2Model

  }
}
