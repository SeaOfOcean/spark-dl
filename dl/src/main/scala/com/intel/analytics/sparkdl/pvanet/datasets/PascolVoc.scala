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

package com.intel.analytics.sparkdl.pvanet.datasets

import com.intel.analytics.sparkdl.pvanet.AnchorGenerator
import scopt.OptionParser

object PascolVoc {

  case class PascolVocLocalParam(folder: String = "/home/xianyan/objectRelated/VOCdevkit")

  private val parser = new OptionParser[PascolVocLocalParam]("Spark-DL PascolVoc Local Example") {
    head("Spark-DL PascolVoc Local Example")
    opt[String]('f', "folder")
      .text("where you put the PascolVoc data")
      .action((x, c) => c.copy(folder = x))
  }

  def main(args: Array[String]) {
    parser.parse(args, new PascolVocLocalParam()).map(param => {
      val year = "2007"
      val validationDataSource = new PascolVocDataSource(year, "test", param.folder, false)
      val trainDataSource = new PascolVocDataSource(year, imageSet = "test", param.folder, false)
      val imageScaler = new ImageScalerAndMeanSubstractor(trainDataSource)
      val toTensor = new ToTensor(batchSize = 1)

      val data = trainDataSource ++ imageScaler ++ AnchorGenerator ++ toTensor
    })
  }
}
