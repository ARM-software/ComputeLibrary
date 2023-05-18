/*
 * Copyright (c) 2016-2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ARM_COMPUTE_CLFUNCTIONS_H
#define ARM_COMPUTE_CLFUNCTIONS_H

/* Header regrouping all the CL functions */
#include "arm_compute/runtime/CL/functions/CLActivationLayer.h"
#include "arm_compute/runtime/CL/functions/CLArgMinMaxLayer.h"
#include "arm_compute/runtime/CL/functions/CLBatchNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLBatchToSpaceLayer.h"
#include "arm_compute/runtime/CL/functions/CLBitwiseAnd.h"
#include "arm_compute/runtime/CL/functions/CLBitwiseNot.h"
#include "arm_compute/runtime/CL/functions/CLBitwiseOr.h"
#include "arm_compute/runtime/CL/functions/CLBitwiseXor.h"
#include "arm_compute/runtime/CL/functions/CLBoundingBoxTransform.h"
#include "arm_compute/runtime/CL/functions/CLCast.h"
#include "arm_compute/runtime/CL/functions/CLChannelShuffleLayer.h"
#include "arm_compute/runtime/CL/functions/CLComparison.h"
#include "arm_compute/runtime/CL/functions/CLConcatenateLayer.h"
#include "arm_compute/runtime/CL/functions/CLConv3D.h"
#include "arm_compute/runtime/CL/functions/CLConvertFullyConnectedWeights.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLCopy.h"
#include "arm_compute/runtime/CL/functions/CLCrop.h"
#include "arm_compute/runtime/CL/functions/CLCropResize.h"
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLDeconvolutionLayerUpsample.h"
#include "arm_compute/runtime/CL/functions/CLDepthConvertLayer.h"
#include "arm_compute/runtime/CL/functions/CLDepthToSpaceLayer.h"
#include "arm_compute/runtime/CL/functions/CLDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLDequantizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLDirectConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLDirectDeconvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseOperations.h"
#include "arm_compute/runtime/CL/functions/CLElementwiseUnaryLayer.h"
#include "arm_compute/runtime/CL/functions/CLFFT1D.h"
#include "arm_compute/runtime/CL/functions/CLFFT2D.h"
#include "arm_compute/runtime/CL/functions/CLFFTConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLFill.h"
#include "arm_compute/runtime/CL/functions/CLFlattenLayer.h"
#include "arm_compute/runtime/CL/functions/CLFloor.h"
#include "arm_compute/runtime/CL/functions/CLFullyConnectedLayer.h"
#include "arm_compute/runtime/CL/functions/CLFuseBatchNormalization.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/functions/CLGEMMConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMDeconvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "arm_compute/runtime/CL/functions/CLGather.h"
#include "arm_compute/runtime/CL/functions/CLGenerateProposalsLayer.h"
#include "arm_compute/runtime/CL/functions/CLIndirectConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLInstanceNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLL2NormalizeLayer.h"
#include "arm_compute/runtime/CL/functions/CLLSTMLayer.h"
#include "arm_compute/runtime/CL/functions/CLLSTMLayerQuantized.h"
#include "arm_compute/runtime/CL/functions/CLLogicalAnd.h"
#include "arm_compute/runtime/CL/functions/CLLogicalNot.h"
#include "arm_compute/runtime/CL/functions/CLLogicalOr.h"
#include "arm_compute/runtime/CL/functions/CLMatMul.h"
#include "arm_compute/runtime/CL/functions/CLMaxUnpoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLMeanStdDevNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLNormalizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLNormalizePlanarYUVLayer.h"
#include "arm_compute/runtime/CL/functions/CLPReluLayer.h"
#include "arm_compute/runtime/CL/functions/CLPadLayer.h"
#include "arm_compute/runtime/CL/functions/CLPermute.h"
#include "arm_compute/runtime/CL/functions/CLPixelWiseMultiplication.h"
#include "arm_compute/runtime/CL/functions/CLPooling3dLayer.h"
#include "arm_compute/runtime/CL/functions/CLPoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLPriorBoxLayer.h"
#include "arm_compute/runtime/CL/functions/CLQLSTMLayer.h"
#include "arm_compute/runtime/CL/functions/CLQuantizationLayer.h"
#include "arm_compute/runtime/CL/functions/CLRNNLayer.h"
#include "arm_compute/runtime/CL/functions/CLROIAlignLayer.h"
#include "arm_compute/runtime/CL/functions/CLROIPoolingLayer.h"
#include "arm_compute/runtime/CL/functions/CLRange.h"
#include "arm_compute/runtime/CL/functions/CLReduceMean.h"
#include "arm_compute/runtime/CL/functions/CLReductionOperation.h"
#include "arm_compute/runtime/CL/functions/CLReorgLayer.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "arm_compute/runtime/CL/functions/CLReverse.h"
#include "arm_compute/runtime/CL/functions/CLScale.h"
#include "arm_compute/runtime/CL/functions/CLSelect.h"
#include "arm_compute/runtime/CL/functions/CLSlice.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/runtime/CL/functions/CLSpaceToBatchLayer.h"
#include "arm_compute/runtime/CL/functions/CLSpaceToDepthLayer.h"
#include "arm_compute/runtime/CL/functions/CLSplit.h"
#include "arm_compute/runtime/CL/functions/CLStackLayer.h"
#include "arm_compute/runtime/CL/functions/CLStridedSlice.h"
#include "arm_compute/runtime/CL/functions/CLTile.h"
#include "arm_compute/runtime/CL/functions/CLTranspose.h"
#include "arm_compute/runtime/CL/functions/CLUnstack.h"
#include "arm_compute/runtime/CL/functions/CLWinogradConvolutionLayer.h"

#endif /* ARM_COMPUTE_CLFUNCTIONS_H */
