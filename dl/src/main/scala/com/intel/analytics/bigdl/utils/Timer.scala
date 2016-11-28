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

package com.intel.analytics.bigdl.utils

class Timer {
  var totalTime: Long = 0
  var calls: Int = 0
  var startTime: Long = 0
  var diff: Long = 0
  var averageTime: Long = 0

  def tic(): Unit = {
    startTime = System.nanoTime()
  }

  def toc(average: Boolean = true): Long = {
    diff = System.nanoTime() - startTime
    totalTime += diff
    calls += 1
    averageTime = totalTime / calls
    if (average) averageTime else diff
  }
}
