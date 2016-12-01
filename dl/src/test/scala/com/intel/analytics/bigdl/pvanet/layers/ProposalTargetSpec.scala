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
import com.intel.analytics.bigdl.pvanet.model.VggParam
import com.intel.analytics.bigdl.pvanet.utils.FileUtil
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.FlatSpec

class ProposalTargetSpec extends FlatSpec {

  val param = new VggParam(true)
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
  val proposalTarget = new ProposalTarget[Float](21, param)
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

    val targets = proposalTarget.computeTargets(convert(exRois, Float),
      convert(gtRois, Float), labels)

    TestUtil.assertMatrixEqualFD(targets, expected, 1e-4)
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

    val targets = proposalTarget.computeTargets(convert(exRois, Float),
      convert(gtRois, Float), labels)

    TestUtil.assertMatrixEqualFD(targets, expected, 1e-4)
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

    val (r1, r2) = proposalTarget.getBboxRegressionLabels(convert(data, Float), 6)

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
    TestUtil.assertMatrixEqualFD(r1, convert(expected1, Double), 1e-6)
    TestUtil.assertMatrixEqualFD(r2, convert(expected2, Double), 1e-6)

  }

  // the following test code should fix keep_inds to Array(13 28  4) since
  // there is random shuffle when getting the fg_inds and bg_inds
  //  it should "get right proposal target " in {
  //    val data1 = DenseMatrix(
  // (0.0, 0.543404941791, 0.278369385094, 0.424517590749, 0.84477613232),
  //      (0.0, 0.00471885619097, 0.121569120783, 0.670749084727, 0.825852755105),
  //      (0.0, 0.136706589685, 0.575093329427, 0.891321954312, 0.209202122117),
  //      (0.0, 0.18532821955, 0.108376890464, 0.219697492625, 0.978623784707),
  //      (0.0, 0.811683149089, 0.171941012733, 0.816224748726, 0.274073747042),
  //      (0.0, 0.431704183663, 0.940029819622, 0.817649378777, 0.336111950121),
  //      (0.0, 0.175410453742, 0.37283204629, 0.00568850735257, 0.252426353445),
  //      (0.0, 0.795662508473, 0.0152549712463, 0.598843376928, 0.603804539043),
  //      (0.0, 0.105147685412, 0.381943444943, 0.0364760565926, 0.890411563442),
  //      (0.0, 0.980920857012, 0.059941988818, 0.890545944729, 0.5769014994),
  //      (0.0, 0.742479689098, 0.630183936475, 0.581842192399, 0.0204391320269),
  //      (0.0, 0.210026577673, 0.544684878179, 0.769115171106, 0.250695229138),
  //      (0.0, 0.285895690407, 0.852395087841, 0.975006493607, 0.884853293491),
  //      (0.0, 0.359507843937, 0.598858945876, 0.354795611657, 0.340190215371),
  //      (0.0, 0.178080989506, 0.237694208624, 0.0448622824608, 0.505431429636),
  //      (0.0, 0.376252454297, 0.592805400976, 0.629941875587, 0.142600314446),
  //      (0.0, 0.933841299466, 0.946379880809, 0.602296657731, 0.387766280327),
  //      (0.0, 0.363188004109, 0.204345276869, 0.276765061396, 0.246535881204),
  //      (0.0, 0.17360800174, 0.966609694487, 0.957012600353, 0.597973684329),
  //      (0.0, 0.73130075306, 0.340385222837, 0.0920556033772, 0.463498018937),
  //      (0.0, 0.508698893238, 0.0884601730029, 0.528035223318, 0.992158036511),
  //      (0.0, 0.395035931758, 0.335596441719, 0.805450537329, 0.754348994582),
  //      (0.0, 0.313066441589, 0.634036682962, 0.540404575301, 0.29679375088),
  //      (0.0, 0.110787901182, 0.312640297876, 0.456979130049, 0.658940070226),
  //      (0.0, 0.254257517818, 0.641101258701, 0.200123607218, 0.657624805529),
  //      (0.0, 0.77828921545, 0.779598398611, 0.610328153209, 0.309000348524),
  //      (0.0, 0.697734907513, 0.859618295729, 0.625323757757, 0.98240782961),
  //      (0.0, 0.976500127016, 0.166694131199, 0.0231781364784, 0.160744548507),
  //      (0.0, 0.923496825259, 0.95354984988, 0.210978418718, 0.360525250815),
  //      (0.0, 0.549375261628, 0.271830849177, 0.460601621075, 0.696161564823))
  //
  //    val data2 = DenseMatrix(
  // (0.500355896675, 0.716070990564, 0.52595593623, 0.00139902311904, 0.39470028669),
  //      (0.492166969901, 0.402880331379, 0.354298300106, 0.500614319443, 0.445176628831),
  //      (0.0904327881964, 0.273562920027, 0.943477097743, 0.0265446413339, 0.0399986896407),
  //      (0.28314035972, 0.582344170217, 0.990892802925, 0.992642237403, 0.993117372481),
  //      (0.110048330967, 0.664481445964, 0.523986834488, 0.173149909809, 0.942960244915),
  //      (0.241860085976, 0.998932268843, 0.58269381515, 0.183279000631, 0.386845421918),
  //      (0.189673528912, 0.410770673025, 0.594680068902, 0.716586093128, 0.486891482369),
  //      (0.309589817767, 0.577441372828, 0.441707819569, 0.359678102601, 0.321331932009),
  //      (0.208207240196, 0.451258624062, 0.491842910264, 0.899076314794, 0.729360461029),
  //      (0.77008977292, 0.375439247562, 0.343739535235, 0.655035205999, 0.71103799321),
  //      (0.113537575219, 0.133028689374, 0.456039057606, 0.159736230159, 0.961641903775),
  //      (0.837615744862, 0.520160687038, 0.218272257728, 0.134918722532, 0.979070345484),
  //      (0.707043495689, 0.859975556946, 0.387172627829, 0.250834019832, 0.299438018945),
  //      (0.856895528405, 0.472983990568, 0.663277047016, 0.805728607437, 0.25298050465),
  //      (0.0795734389703, 0.732760605016, 0.961397477504, 0.953804734168, 0.490499051884),
  //      (0.632192064433, 0.732995019838, 0.902409503248, 0.162246918748, 0.405881322368),
  //      (0.417090735584, 0.695591028292, 0.424847237925, 0.858114226051, 0.846932479609),
  //      (0.0701991139087, 0.301752413484, 0.97962368103, 0.035626996553, 0.492392646999),
  //      (0.952376853014, 0.810573758529, 0.294330441296, 0.596233518518, 0.4311778523),
  //      (0.592397502989, 0.89375210472, 0.554021189772, 0.492866507345, 0.31927045719),
  //      (0.263365783051, 0.542280613536, 0.082264523932, 0.635636709825, 0.796405225186),
  //      (0.954747505431, 0.684624271693, 0.488293166805, 0.485414310184, 0.966692920583),
  //      (0.211347887497, 0.411648138178, 0.989665576779, 0.0284118567133, 0.701326514094),
  //      (0.0251715638848, 0.320881726087, 0.0735270618656, 0.0608845643466, 0.111406316704),
  //      (0.169268908145, 0.627686279501, 0.43839309464, 0.830903764604, 0.239792189564),
  //      (0.19005270792, 0.711899658583, 0.858294925327, 0.559055885596, 0.704420408289),
  //      (0.605112035518, 0.559217283268, 0.860394190908, 0.91975535915, 0.849607325759),
  //      (0.254466535494, 0.877555542287, 0.435130190092, 0.729494343965, 0.412640767539),
  //      (0.190836045811, 0.706019519956, 0.24063282093, 0.851324426833, 0.824102289259),
  //      (0.525211786614, 0.386340794306, 0.590880790735, 0.137523614908, 0.808270407892))
  //
  //    val input = new Table
  //    val fd1 = convert(data1, Float)
  //    val fd2 = convert(data2, Float)
  //    input.insert(matrix2tensor(fd1))
  //    input.insert(matrix2tensor(fd2))
  //    val out = proposalTarget.forward(input)
  //
  //    val expectedO1 = DenseMatrix(
  // (0.0, 0.359507843937, 0.598858945876, 0.354795611657, 0.340190215371),
  //      (0.0, 0.923496825259, 0.95354984988, 0.210978418718, 0.360525250815),
  //      (0.0, 0.811683149089, 0.171941012733, 0.816224748726, 0.274073747042))
  //    val labels = DenseMatrix(0.321331932009, 0.0, 0.0)
  //    val expectedO3 = DenseMatrix(
  //      (0.0, 0.0, 0.0),
  //      (0.0185847, 0.0, 0.0),
  //      (-0.0013015, 0.0, 0.0),
  //      (0.128814, 0.0, 0.0),
  //      (0.0537098, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0)).t
  //    val expectedO4 = DenseMatrix((0.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0)).t
  //
  //    val expectedO5 = DenseMatrix((0.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (1.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0),
  //      (0.0, 0.0, 0.0)).t
  //    TestUtil.assertMatrixEqualTM(out(1).asInstanceOf[Tensor[Float]], expectedO1, 1e-6)
  //    TestUtil.assertMatrixEqualTM(out(2).asInstanceOf[Tensor[Float]], labels, 1e-6)
  //    TestUtil.assertMatrixEqualTM(out(3).asInstanceOf[Tensor[Float]], expectedO3, 1e-6)
  //    TestUtil.assertMatrixEqualTM(out(4).asInstanceOf[Tensor[Float]], expectedO4, 1e-6)
  //    TestUtil.assertMatrixEqualTM(out(5).asInstanceOf[Tensor[Float]], expectedO5, 1e-6)

  //  }

  def matrix2tensor(mat: DenseMatrix[Float]): Tensor[Float] = {
    val out = Tensor[Float]().resize(mat.rows, mat.cols)
    for (i <- 0 until mat.rows) {
      for (j <- 0 until mat.cols) {
        out.setValue(i + 1, j + 1, mat(i, j))
      }
    }
    out
  }

}
