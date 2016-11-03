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

import com.intel.analytics.sparkdl.pvanet.Roidb.ImageWithRoi

abstract class Imdb {
  var name = ""
  var classes = Array[String]()
  var imageIndex = List[String]()

  var _roidb = Array[ImageWithRoi]()

  /**
    * A roidb is a list of dictionaries, each with the following keys:
    * boxes
    * gt_overlaps
    * gt_classes
    * flipped
    */
  def roidb(): Array[ImageWithRoi] = {
    if (_roidb.length > 0) return _roidb
    _roidb = getGroundTruth()
    return _roidb

  }

  def getGroundTruth(): Array[ImageWithRoi]


  def numClasses = classes.length


  def numImages() = imageIndex.length

  def imagePathAt(i: Int): String

  def appendFlippedImages() = ???
}

