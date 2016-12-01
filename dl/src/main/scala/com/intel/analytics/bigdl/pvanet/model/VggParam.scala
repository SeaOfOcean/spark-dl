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

package com.intel.analytics.bigdl.pvanet.model

class VggParam(isTrain: Boolean = false) extends FasterRcnnParam(isTrain) {
  override val anchorScales = Array[Float](8, 16, 32)
  override val anchorRatios = Array[Float](0.5f, 1.0f, 2.0f)
  override val anchorNum = 9

  override val RPN_PRE_NMS_TOP_N = if (isTrain) 12000 else 6000
  override val RPN_POST_NMS_TOP_N = if(isTrain) 2000 else 300
}
