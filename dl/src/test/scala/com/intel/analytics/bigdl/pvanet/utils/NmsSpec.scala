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

package com.intel.analytics.bigdl.pvanet.utils

import breeze.linalg.{DenseMatrix, convert}
import com.intel.analytics.bigdl.pvanet.TestUtil
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

import scala.io.Source

class NmsSpec extends FlatSpec with Matchers {
  def loadDataFromFile(fileName: String, sizes: Array[Int]): Tensor[Float] = {
    val lines = Source.fromFile(fileName).getLines().toArray.map(x => x.toFloat)
    Tensor(Storage(lines)).resize(sizes)
  }

  val classLoader = getClass().getClassLoader()

  behavior of "NmsSpec"

  it should "nms" in {
    val dets = DenseMatrix(
      (0.771320643267, 0.0207519493594, 0.633648234926, 0.748803882539, 0.498507012303),
      (0.224796645531, 0.19806286476, 0.760530712199, 0.169110836563, 0.088339814174),
      (0.685359818368, 0.953393346195, 0.00394826632791, 0.512192263386, 0.812620961652),
      (0.612526066829, 0.721755317432, 0.291876068171, 0.917774122513, 0.714575783398),
      (0.542544368011, 0.142170047602, 0.373340760051, 0.674133615066, 0.441833174423),
      (0.434013993333, 0.617766978469, 0.513138242554, 0.650397181931, 0.601038953405),
      (0.805223196833, 0.521647152394, 0.908648880809, 0.319236088989, 0.0904593492709),
      (0.300700056636, 0.113984361864, 0.828681326308, 0.0468963193892, 0.626287148311),
      (0.547586155919, 0.81928699567, 0.198947539679, 0.856850302458, 0.351652639432),
      (0.75464769153, 0.29596170688, 0.883936479561, 0.325511637832, 0.165015897719),
      (0.392529243947, 0.0934603745587, 0.821105657837, 0.151152019643, 0.384114448692),
      (0.944260712239, 0.987625474902, 0.456304547095, 0.826122843843, 0.251374134207),
      (0.597371648231, 0.902831760332, 0.534557948802, 0.590201362985, 0.0392817672254),
      (0.357181758635, 0.079613090156, 0.305459918343, 0.330719311982, 0.773830296211),
      (0.03995920869, 0.429492178432, 0.314926871843, 0.636491143068, 0.34634715008),
      (0.043097356205, 0.879915174518, 0.763240587144, 0.878096642725, 0.417509143839),
      (0.605577564394, 0.513466627408, 0.597836647963, 0.262215661132, 0.300871308941),
      (0.0253997820501, 0.303062560651, 0.242075875404, 0.557578188663, 0.565507019888),
      (0.475132247415, 0.29279797629, 0.0642510606948, 0.978819145758, 0.339707843638),
      (0.495048630882, 0.977080725923, 0.440773824901, 0.318272805479, 0.519796985875),
      (0.578136429882, 0.8539337505, 0.068097273538, 0.464530807779, 0.781949118619),
      (0.718602810382, 0.586021980053, 0.0370944132344, 0.350656391283, 0.563190684493),
      (0.299729872425, 0.512334153274, 0.673466925285, 0.159193733378, 0.050477670154),
      (0.337815887065, 0.108063772779, 0.178902808571, 0.885827096168, 0.365364971214),
      (0.21876934918, 0.752496170219, 0.106879584394, 0.744603240776, 0.46978529344),
      (0.598255671279, 0.147620192285, 0.184034822093, 0.645072126468, 0.0486280062634),
      (0.248612507803, 0.542408516228, 0.22677334327, 0.381411534905, 0.922232786904),
      (0.925356872868, 0.566749924575, 0.533470884989, 0.0148600246332, 0.977899263402),
      (0.573028904033, 0.791756996277, 0.561557360276, 0.877335241565, 0.584195828531),
      (0.708849826369, 0.148533451356, 0.428450738968, 0.693890066342, 0.104619744523))

    val dets2 = convert(dets, Float)

    val dets2copy = dets2.copy

    val keep = Nms.nms(dets2, 0.1f)

    TestUtil.assertMatrixEqual(dets2copy, dets2, 1e-9f)

    val expected = Array(27, 2, 11)
    (expected zip keep).foreach(x => assert(x._1 == x._2))


    val keep2 = Nms.nms(dets2, 0.3f)

    val expected2 = Array(27, 26, 2, 7, 28, 19)
    (expected2 zip keep2).foreach(x => assert(x._1 == x._2))

    val tensor2 = Tensor(dets2)

    val keepT = Nms.nms(tensor2.contiguous(), 0.1f)
    keepT should be(expected.map(x => x + 1))
    Nms.nms(tensor2.contiguous(), 0.3f) should be(expected2.map(x => x + 1))

    val det3 = loadDataFromFile(classLoader.getResource("pvanet/nms.dat").getFile, Array(1009, 5))

    val keep3 = Nms.nms(det3.toBreezeMatrix(), 0.7f)
    val expected3 = Array(0, 4, 6, 232)
    (expected3 zip keep3).foreach(x => assert(x._1 == x._2))

    Nms.nms(det3, 0.7f) should be(expected3.map(x => x + 1))
  }

}
