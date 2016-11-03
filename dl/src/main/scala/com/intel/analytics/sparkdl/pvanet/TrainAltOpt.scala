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

  def train_rpn(imdbName: String, initModel: String, maxIters: Int): String = ???

  def rpn_generate(imdbName: String, rpnModelPath: String) = ???

  def train_fast_rcnn(imdbName: String, initModel: String, maxIters: Int, rpn_proposals: String) = ???


  def main(args: Array[String]): Unit = {
    val imdbName = "voc_2007"
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 RPN, init from ImageNet model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpn_stage1_model = train_rpn(imdbName, "pretrained_path", 10)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 RPN, generate proposals")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpn_stage1_proposals = rpn_generate(imdbName, rpn_stage1_model)
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    val fast_rcnn_stage1_model = train_fast_rcnn(imdbName, "pretrained_path", 10, rpn_stage1_proposals)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 RPN, init from stage 1 Fast R-CNN model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    val rpn_stage2_model = train_rpn(imdbName, fast_rcnn_stage1_model, 10)

    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 RPN, generate proposals")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val rpn_stage2_proposals = rpn_generate(imdbName, rpn_stage2_model)



    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    println("Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model")
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    val fast_rcnn_stage2_model = train_fast_rcnn(imdbName, rpn_stage2_model, 10, rpn_stage2_proposals)
    val finalModelPath = fast_rcnn_stage2_model

  }
}
