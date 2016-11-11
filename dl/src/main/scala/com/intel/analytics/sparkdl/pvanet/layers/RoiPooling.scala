package com.intel.analytics.sparkdl.pvanet.layers

import breeze.linalg.{max, min}
import breeze.numerics.{ceil, floor, round}
import com.intel.analytics.sparkdl.nn.Module
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.Table

import scala.reflect.ClassTag

class RoiPooling[@specialized(Float, Double) T: ClassTag](val pooled_w: Int, val pooled_h: Int, val spatial_scale: T)
                                                         (implicit ev: TensorNumeric[T]) extends Module[Table, Tensor[T], T] {
  @transient var channels = 0
  @transient var height = 0
  @transient var width = 0

  override def updateOutput(input: Table): Tensor[T] = {
    assert(input.length() == 2, "there must have two tensors in the table")
    val data = input(1).asInstanceOf[Tensor[T]]
    val rois = input(2).asInstanceOf[Tensor[T]]
    assert(rois.size().length > 1 && rois.size()(1) == 5, "roi input shape should be (R, 5)")
    assert(rois.size()(0) * rois.size()(1) == rois.nElement(), "roi input shape should be (R, 5)")
    channels = data.size()(1)
    height = data.size()(2)
    width = data.size()(3)

    val numRois = rois.size()(0)
    val batchSize = data.size()(0)
    val roiData = rois.storage().array()
    output = Tensor(numRois, channels, pooled_h, pooled_w)

    output.fill(ev.fromType[Double](-Double.MaxValue))
    var outputData = output.storage().array()
    var argmax = Tensor()
    argmax.resizeAs(output)
    argmax.fill(ev.fromType(-1))
    var argmax_data = argmax.storage().array()

    val bottom_data = data.storage().array()
    val bottom_rois = rois.storage().array()

    //    val topData = output.storage().array()

    var topDataIndex = 0
    var argmaxIndex = 0

    var bottomRoisIndex = 0
    for (n <- 0 until numRois) {
      val roiBatchInd = roiData(bottomRoisIndex)
      def roundRoi(ind: Int) = round(ev.toType[Double](roiData(bottomRoisIndex + ind)) * ev.toType[Double]
        (spatial_scale))
        .toInt
      val roi_start_w = roundRoi(1)
      val roi_start_h = roundRoi(2)
      val roi_end_w = roundRoi(3)
      val roi_end_h = roundRoi(4)
      assert(ev.isGreaterEq(roiBatchInd, ev.fromType(0)))
      assert(ev.isGreater(ev.fromType(batchSize), roiBatchInd))

      val roiHeight = max(roi_end_h - roi_start_h + 1, 1)
      val roiWidth = max(roi_end_w - roi_start_w + 1, 1)
      val binSizeH = ev.divide(ev.fromType[Double](roiHeight), ev.fromType[Int](pooled_h))
      val binSizeW = ev.divide(ev.fromType[Double](roiWidth), ev.fromType[Int](pooled_w))
      var batchDataIndex = offset(ev.toType[Int](roiBatchInd), sizes = data.size())
      for (c <- 0 until channels) {
        for (ph <- 0 until pooled_h) {
          for (pw <- 0 until pooled_w) {
            // Compute pooling region for this output unit:
            //  start (included) = floor(ph * roi_height / pooled_height_)
            //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
            var hstart = floor(ph * ev.toType[Int](binSizeH))
            var wstart = floor(pw * ev.toType[Int](binSizeW))
            var hend = ceil((ph + 1) * ev.toType[Int](binSizeH))
            var wend = ceil((pw + 1) * ev.toType[Int](binSizeW))

            hstart = min(max(hstart + roi_start_h, 0), height)
            hend = min(max(hend + roi_start_h, 0), height)
            wstart = min(max(wstart + roi_start_w, 0), width)
            wend = min(max(wend + roi_start_w, 0), width)

            val is_empty = (hend <= hstart) || (wend <= wstart)

            val pool_index = ph * pooled_w + pw
            if (is_empty) {
              outputData(topDataIndex + pool_index) = ev.fromType(0)
              argmax_data(argmaxIndex + pool_index) = ev.fromType(-1)
            }

            for (h <- hstart until hend) {
              for (w <- wstart until wend) {
                val index = h * width + w
                if (ev.isGreater(bottom_data(batchDataIndex + index), outputData(topDataIndex + pool_index))) {
                  outputData(topDataIndex + pool_index) = bottom_data(batchDataIndex + index)
                  argmax_data(argmaxIndex + pool_index) = ev.fromType(index)
                }
              }
            }
          }
        }
        // Increment all data pointers by one channel
        batchDataIndex += offset(0, 1, sizes = data.size())
        topDataIndex += offset(0, 1, sizes = output.size())
        argmaxIndex += offset(0, 1, sizes = argmax.size())
      }
      bottomRoisIndex += offset(1, sizes = rois.size())

    }
    output
  }

  def offset(n: Int, c: Int = 0, h: Int = 0, w: Int = 0, sizes: Array[Int]): Int = {
    assert(sizes.length == 2 || sizes.length >= 4)
    if (sizes.length == 2) ((n * sizes(1) + c) + h) + w
    else ((n * sizes(1) + c) * sizes(2) + h) * sizes(3) + w
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[T]): Table = ???
}
