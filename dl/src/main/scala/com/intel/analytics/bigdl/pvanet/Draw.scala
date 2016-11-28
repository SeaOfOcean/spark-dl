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

package com.intel.analytics.bigdl.pvanet

import java.awt.image.BufferedImage
import java.awt.{BasicStroke, Color, Font, Graphics2D}
import java.io.File
import javax.imageio.ImageIO

import breeze.linalg.DenseMatrix

object Draw {
  def vis(imgPath: String, clsname: String, dets: DenseMatrix[Float],
    savePath: String, thresh: Float = 0.3f): Unit = {
    var img: BufferedImage = null
    var g2d: Graphics2D = null

    def loadImage = {
      img = ImageIO.read(new File(imgPath))
      g2d = img.createGraphics
      val font = new Font("Helvetica", Font.PLAIN, 14);
      g2d.setFont(font)
      g2d.setStroke(new BasicStroke(3))
    }

    for (i <- 0 until Math.min(10, dets.rows)) {
      val bbox = dets(i, 0 until 4)
      val score = dets(i, 4)
      if (score > thresh) {
        if (g2d == null) {
          loadImage
        }
        draw(g2d, bbox(0).toInt, bbox(1).toInt, bbox(2).toInt - bbox(0).toInt,
          bbox(3).toInt - bbox(1).toInt, s"$clsname ${"%.3f".format(score)}")
      }
    }
    if (g2d != null) {
      ImageIO.write(img, savePath.substring(savePath.lastIndexOf(".") + 1), new File(savePath))
      println(savePath + " is saved")
      g2d.dispose
    }
  }

  def draw(img: Graphics2D, x1: Int, y1: Int, width: Int, height: Int, title: String): Unit = {
    img.setColor(Color.RED)
    img.drawRect(x1, y1, width, height)
    val bgColor = new Color(0, 0, 1, 127)

    val fm = img.getFontMetrics()
    val rect = fm.getStringBounds(title, img)

    img.setColor(bgColor)
    img.fillRect(x1, y1 - 2 - fm.getAscent(),
      rect.getWidth().toInt,
      rect.getHeight().toInt)
    img.setColor(Color.WHITE)
    img.drawString(title, x1, y1 - 2);
  }
}
