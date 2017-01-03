/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

class ImageToTensor(batchSize: Int = 1) extends Transformer[ImageWithRoi, Tensor[Float]] {
  require(batchSize == 1)

  private val featureTensor: Tensor[Float] = Tensor[Float]()

  def apply(imgWithRoi: ImageWithRoi): Tensor[Float] = {
    assert(batchSize == 1)
    val img = imgWithRoi.scaledImage
    featureTensor.set(Storage[Float](imgWithRoi.scaledImage.content),
      storageOffset = 1, sizes = Array(batchSize, img.height(), img.width(), 3))
    featureTensor.transpose(2, 3).transpose(2, 4).contiguous()
  }

  override def apply(prev: Iterator[ImageWithRoi]): Iterator[Tensor[Float]] = {
    new Iterator[Tensor[Float]] {


      override def hasNext: Boolean = prev.hasNext

      override def next(): Tensor[Float] = {
        if (prev.hasNext) {
          val imgWithRoi = prev.next()
          apply(imgWithRoi)
        } else {
          null
        }
      }
    }
  }
}

object ImageToTensor {
  val imgToTensor = new ImageToTensor(1)

  def apply(img: ImageWithRoi): Tensor[Float] = imgToTensor.apply(img)
}
