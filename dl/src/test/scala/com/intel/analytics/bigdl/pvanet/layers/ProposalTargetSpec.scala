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

package com.intel.analytics.bigdl.pvanet.layers

import breeze.linalg.{DenseMatrix, convert}
import com.intel.analytics.bigdl.pvanet.TestUtil
import com.intel.analytics.bigdl.pvanet.model.{Phase, VggParam}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.FlatSpec

class ProposalTargetSpec extends FlatSpec {

  val param = new VggParam(Phase.TRAIN)
  val exRois = DenseMatrix((0.543404941791, 0.278369385094, 0.424517590749, 0.84477613232),
    (0.00471885619097, 0.121569120783, 0.670749084727, 0.825852755105),
    (0.136706589685, 0.575093329427, 0.891321954312, 0.209202122117),
    (0.18532821955, 0.108376890464, 0.219697492625, 0.978623784707),
    (0.811683149089, 0.171941012733, 0.816224748726, 0.274073747042),
    (0.431704183663, 0.940029819622, 0.817649378777, 0.336111950121),
    (0.175410453742, 0.37283204629, 0.00568850735257, 0.252426353445),
    (0.795662508473, 0.0152549712463, 0.598843376928, 0.603804539043),
    (0.105147685412, 0.381943444943, 0.0364760565926, 0.890411563442),
    (0.980920857012, 0.059941988818, 0.890545944729, 0.5769014994))
  val gtRois = DenseMatrix((0.742479689098, 0.630183936475, 0.581842192399, 0.0204391320269),
    (0.210026577673, 0.544684878179, 0.769115171106, 0.250695229138),
    (0.285895690407, 0.852395087841, 0.975006493607, 0.884853293491),
    (0.359507843937, 0.598858945876, 0.354795611657, 0.340190215371),
    (0.178080989506, 0.237694208624, 0.0448622824608, 0.505431429636),
    (0.376252454297, 0.592805400976, 0.629941875587, 0.142600314446),
    (0.933841299466, 0.946379880809, 0.602296657731, 0.387766280327),
    (0.363188004109, 0.204345276869, 0.276765061396, 0.246535881204),
    (0.17360800174, 0.966609694487, 0.957012600353, 0.597973684329),
    (0.73130075306, 0.340385222837, 0.0920556033772, 0.463498018937)
  )


  val labels = Array(0.508698893238, 0.0884601730029, 0.528035223318, 0.992158036511,
    0.395035931758, 0.335596441719, 0.805450537329, 0.754348994582, 0.313066441589,
    0.634036682962).map(x => x.toFloat)


  behavior of "ProposalTargetSpec"

  val proposalTarget = new ProposalTarget[Float](param)
  it should "computeTargets without norm correcly" in {
    val expected = DenseMatrix((0.508699, 0.202244, -0.15083, -0.0485428, -1.38974),
      (0.0884602, 0.0911369, -0.0446058, -0.0663423, -0.88127),
      (0.528035, 0.0663603, 0.751411, -0.0380474, 0.487477),
      (0.992158, 0.149501, -0.039554, -0.0385152, -0.925378),
      (0.395036, -0.699306, 0.134789, -0.1475, 0.139986),
      (0.335596, -0.0877232, -0.682606, -0.100292, 0.327924),
      (0.805451, 0.816015, 0.402963, -0.216791, -0.68954),
      (0.754349, -0.469728, -0.0529346, 0.128788, -0.421497),
      (0.313066, 0.53096, 0.0968626, 0.649668, -0.870967),
      (0.634037, -0.576122, 0.0550574, -0.924834, -0.300604))
    param.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = false
    val targets = proposalTarget.computeTargets(Tensor(convert(exRois, Float)),
      Tensor(convert(gtRois, Float)), Tensor(Storage(labels)))

    TestUtil.assertMatrixEqualTM(targets, expected, 1e-4)
  }

  it should "computeTargets with norm correcly" in {
    val expected = DenseMatrix((0.508699, 2.02244, -1.5083, -0.242714, -6.94869),
      (0.0884602, 0.911369, -0.446058, -0.331711, -4.40635),
      (0.528035, 0.663603, 7.51411, -0.190237, 2.43739),
      (0.992158, 1.49501, -0.39554, -0.192576, -4.62689),
      (0.395036, -6.99306, 1.34789, -0.7375, 0.699932),
      (0.335596, -0.877232, -6.82606, -0.501458, 1.63962),
      (0.805451, 8.16015, 4.02963, -1.08396, -3.4477),
      (0.754349, -4.69728, -0.529346, 0.643939, -2.10748),
      (0.313066, 5.3096, 0.968626, 3.24834, -4.35484),
      (0.634037, -5.76122, 0.550574, -4.62417, -1.50302))

    param.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = true

    val targets = proposalTarget.computeTargets(Tensor(convert(exRois, Float)),
      Tensor(convert(gtRois, Float)), Tensor(Storage(labels)))

    TestUtil.assertMatrixEqualTM(targets, expected, 1e-4)
  }


  it should "getBboxRegressionLabels" in {
    val data = DenseMatrix((0, 14, 2, 17, 16),
      (0, 15, 4, 11, 16),
      (3, 9, 2, 12, 4),
      (0, 1, 13, 19, 4),
      (2, 4, 3, 7, 17),
      (4, 15, 1, 14, 7),
      (2, 16, 2, 9, 19),
      (5, 2, 14, 17, 16),
      (2, 15, 7, 13, 6),
      (2, 12, 18, 0, 2))

    val (r1, r2) = proposalTarget.getBboxRegressionLabels(Tensor(convert(data, Float)), 6)

    val expected1 = DenseMatrix(
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 16.0, 0.0, 15.0, 12.0),
      (0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 2.0, 0.0, 7.0, 18.0),
      (0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 9.0, 0.0, 13.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 17.0, 0.0, 19.0, 0.0, 6.0, 2.0),
      (0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0)).t

    val expected2 = DenseMatrix(
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0),
      (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0),
      (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
      (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)).t
    TestUtil.assertMatrixEqualTM(r1, convert(expected1, Double), 1e-6)
    TestUtil.assertMatrixEqualTM(r2, convert(expected2, Double), 1e-6)

  }
// uncomment in ProposalTarget if you want to test with real data
  // fg_inds = FileUtil.loadFeatures("fg_inds_choice").storage().array().map(x => x.toInt + 1)
  // bg_inds = FileUtil.loadFeatures("bg_inds_choice").storage().array().map(x => x.toInt + 1)

//  "ProposalTarget" should "work properly" in {
//    val rpn_rois = FileUtil.loadFeatures("rpn_rois")
//    val gt_boxes = FileUtil.loadFeatures("gt_boxes")
//    val input = new Table
//    input.insert(rpn_rois)
//    input.insert(gt_boxes)
//
//    val proposalTarget2 = new ProposalTarget2[Float](param)
//    proposalTarget2.forward(input)
//    val res2 = proposalTarget2.forward(input)
//
//    val rois = FileUtil.loadFeatures("rois")
//    val labels = FileUtil.loadFeatures("labels")
//    val bboxTarget = FileUtil.loadFeatures("bbox_targets")
//    val bboxInsideWeights = FileUtil.loadFeatures("bbox_inside_weights")
//    val bboxOutsideWeights = FileUtil.loadFeatures("bbox_outside_weights")
//
//    val actRois = res2(1).asInstanceOf[Tensor[Float]]
//    val actLabels = res2(2).asInstanceOf[Table](1).asInstanceOf[Tensor[Float]]
//    val actBboxTarget = res2(2).asInstanceOf[Table](2)
//      .asInstanceOf[Table](1).asInstanceOf[Tensor[Float]]
//    val actBboxInsideWeights = res2(2).asInstanceOf[Table](2)
//      .asInstanceOf[Table](2).asInstanceOf[Tensor[Float]]
//    val actBboxOutsideWeights = res2(2).asInstanceOf[Table](2)
//      .asInstanceOf[Table](3).asInstanceOf[Tensor[Float]]
//    FileUtil.assertEqual(rois, actRois, "rois", 1e-6)
//    FileUtil.assertEqual(bboxTarget, actBboxTarget, "bboxTarget", 1e-6)
//    FileUtil.assertEqual(bboxInsideWeights, actBboxInsideWeights, "bboxInsideWeights", 1e-6)
//    FileUtil.assertEqual(labels, actLabels, "labels", 1e-6)
//    FileUtil.assertEqual(bboxOutsideWeights, actBboxOutsideWeights, "bboxOutsideWeights", 1e-6)
//  }
}
