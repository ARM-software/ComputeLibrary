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
#ifndef ARM_COMPUTE_NEFUNCTIONS_H
#define ARM_COMPUTE_NEFUNCTIONS_H

#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEAddMulAdd.h"
#include "arm_compute/runtime/NEON/functions/NEArgMinMaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticAddition.h"
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"
#include "arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEBatchToSpaceLayer.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseAnd.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseNot.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseOr.h"
#include "arm_compute/runtime/NEON/functions/NEBitwiseXor.h"
#include "arm_compute/runtime/NEON/functions/NEBoundingBoxTransform.h"
#include "arm_compute/runtime/NEON/functions/NECast.h"
#include "arm_compute/runtime/NEON/functions/NEChannelShuffleLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConcatenateLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConv3D.h"
#include "arm_compute/runtime/NEON/functions/NEConvertFullyConnectedWeights.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NECopy.h"
#include "arm_compute/runtime/NEON/functions/NECropResize.h"
#include "arm_compute/runtime/NEON/functions/NEDeconvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthConvertLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthToSpaceLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDequantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDetectionPostProcessLayer.h"
#include "arm_compute/runtime/NEON/functions/NEDirectConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFFT1D.h"
#include "arm_compute/runtime/NEON/functions/NEFFT2D.h"
#include "arm_compute/runtime/NEON/functions/NEFFTConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFill.h"
#include "arm_compute/runtime/NEON/functions/NEFillBorder.h"
#include "arm_compute/runtime/NEON/functions/NEFlattenLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFloor.h"
#include "arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h"
#include "arm_compute/runtime/NEON/functions/NEFuseBatchNormalization.h"
#include "arm_compute/runtime/NEON/functions/NEGEMM.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConv2d.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/NEON/functions/NEGEMMLowpOutputStage.h"
#include "arm_compute/runtime/NEON/functions/NEGather.h"
#include "arm_compute/runtime/NEON/functions/NEGenerateProposalsLayer.h"
#include "arm_compute/runtime/NEON/functions/NEInstanceNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h"
#include "arm_compute/runtime/NEON/functions/NELSTMLayer.h"
#include "arm_compute/runtime/NEON/functions/NELSTMLayerQuantized.h"
#include "arm_compute/runtime/NEON/functions/NELogical.h"
#include "arm_compute/runtime/NEON/functions/NEMatMul.h"
#include "arm_compute/runtime/NEON/functions/NEMaxUnpoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NENormalizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPReluLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPadLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPermute.h"
#include "arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h"
#include "arm_compute/runtime/NEON/functions/NEPooling3dLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPriorBoxLayer.h"
#include "arm_compute/runtime/NEON/functions/NEQLSTMLayer.h"
#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"
#include "arm_compute/runtime/NEON/functions/NERNNLayer.h"
#include "arm_compute/runtime/NEON/functions/NEROIAlignLayer.h"
#include "arm_compute/runtime/NEON/functions/NEROIPoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NERange.h"
#include "arm_compute/runtime/NEON/functions/NEReduceMean.h"
#include "arm_compute/runtime/NEON/functions/NEReductionOperation.h"
#include "arm_compute/runtime/NEON/functions/NEReorderLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReorgLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReshapeLayer.h"
#include "arm_compute/runtime/NEON/functions/NEReverse.h"
#include "arm_compute/runtime/NEON/functions/NEScale.h"
#include "arm_compute/runtime/NEON/functions/NESelect.h"
#include "arm_compute/runtime/NEON/functions/NESlice.h"
#include "arm_compute/runtime/NEON/functions/NESoftmaxLayer.h"
#include "arm_compute/runtime/NEON/functions/NESpaceToBatchLayer.h"
#include "arm_compute/runtime/NEON/functions/NESpaceToDepthLayer.h"
#include "arm_compute/runtime/NEON/functions/NESplit.h"
#include "arm_compute/runtime/NEON/functions/NEStackLayer.h"
#include "arm_compute/runtime/NEON/functions/NEStridedSlice.h"
#include "arm_compute/runtime/NEON/functions/NETile.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"
#include "arm_compute/runtime/NEON/functions/NEUnstack.h"
#include "arm_compute/runtime/NEON/functions/NEWinogradConvolutionLayer.h"

#endif /* ARM_COMPUTE_NEFUNCTIONS_H */
