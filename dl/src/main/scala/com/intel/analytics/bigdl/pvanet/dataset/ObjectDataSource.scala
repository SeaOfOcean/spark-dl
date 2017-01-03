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

package com.intel.analytics.bigdl.pvanet.dataset

import java.util.concurrent.atomic.AtomicInteger

import com.intel.analytics.bigdl.dataset.LocalDataSet
import com.intel.analytics.bigdl.pvanet.model.FasterRcnnParam
import com.intel.analytics.bigdl.utils.RandomGenerator

import scala.util.Random


class ObjectDataSource(val imdb: Imdb, val useFlipped: Boolean = false)
  extends LocalDataSet[Roidb] {

  val roidbs = imdb.getRoidb(useFlipped)

  // permutation of the data index
  var perm: Array[Int] = _

  override def shuffle(): Unit = {
    def shuffleWithAspectGrouping(widths: Array[Int], heights: Array[Int]): Unit = {
      val horz = (widths zip heights).map(x => x._1 >= x._2)
      val vert = horz.map(x => !x)
      val horzInds = horz.zipWithIndex.filter(x => x._1).map(x => x._2)
      val vertInds = vert.zipWithIndex.filter(x => x._1).map(x => x._2)
      val indsArr = (Random.shuffle(horzInds.toSeq) ++ Random.shuffle(vertInds.toSeq)).toArray
      val rowPerm = Random.shuffle(Seq.range(0, indsArr.length / 2))
      rowPerm.zipWithIndex.foreach(r => {
        perm(r._1 * 2) = indsArr(r._2 * 2)
        perm(r._1 * 2 + 1) = indsArr(r._2 * 2 + 1)
      })
    }
    if (FasterRcnnParam.ASPECT_GROUPING) {
      shuffleWithAspectGrouping(imdb.widths, imdb.heights)
    } else {
      RandomGenerator.shuffle(perm)
    }
  }

  /**
   * Get a sequence of data
   *
   * @return
   */
  override def data(looped: Boolean): Iterator[Roidb] = {
    perm = roidbs.indices.toArray
    new Iterator[Roidb] {
      private val index = new AtomicInteger()

      override def hasNext: Boolean = {
        if (looped) {
          true
        } else {
          index.get() < perm.length
        }
      }

      override def next(): Roidb = {
        val curIndex = index.getAndIncrement()
        if (looped || curIndex < perm.length) {
          roidbs(perm(if (looped) curIndex % perm.length else curIndex))
        } else {
          null.asInstanceOf[Roidb]
        }
      }
    }
  }

  /**
   * Return the total size of the data set
   *
   * @return
   */
  override def size(): Long = roidbs.length
}

object ObjectDataSource {

  def apply(name: String, devkitPath: String, useFlipped: Boolean): ObjectDataSource =
    new ObjectDataSource(Imdb.getImdb(name, Some(devkitPath)), useFlipped)
}


class RGBImageOD(protected var data: Array[Float], protected var _width: Int,
  protected var _height: Int) {

  def width(): Int = _width

  def height(): Int = _height

  def content: Array[Float] = data
}
