/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_GRAPH_NODES_H
#define ARM_COMPUTE_GRAPH_NODES_H

#include "arm_compute/graph/nodes/ActivationLayerNode.h"
#include "arm_compute/graph/nodes/ArgMinMaxLayerNode.h"
#include "arm_compute/graph/nodes/BatchNormalizationLayerNode.h"
#include "arm_compute/graph/nodes/BoundingBoxTransformLayerNode.h"
#include "arm_compute/graph/nodes/ChannelShuffleLayerNode.h"
#include "arm_compute/graph/nodes/ConcatenateLayerNode.h"
#include "arm_compute/graph/nodes/ConstNode.h"
#include "arm_compute/graph/nodes/ConvolutionLayerNode.h"
#include "arm_compute/graph/nodes/DeconvolutionLayerNode.h"
#include "arm_compute/graph/nodes/DepthToSpaceLayerNode.h"
#include "arm_compute/graph/nodes/DepthwiseConvolutionLayerNode.h"
#include "arm_compute/graph/nodes/DequantizationLayerNode.h"
#include "arm_compute/graph/nodes/DetectionOutputLayerNode.h"
#include "arm_compute/graph/nodes/DetectionPostProcessLayerNode.h"
#include "arm_compute/graph/nodes/DummyNode.h"
#include "arm_compute/graph/nodes/EltwiseLayerNode.h"
#include "arm_compute/graph/nodes/FlattenLayerNode.h"
#include "arm_compute/graph/nodes/FullyConnectedLayerNode.h"
#include "arm_compute/graph/nodes/FusedConvolutionBatchNormalizationNode.h"
#include "arm_compute/graph/nodes/FusedConvolutionBatchNormalizationWithPostOpsNode.h"
#include "arm_compute/graph/nodes/FusedConvolutionWithPostOpNode.h"
#include "arm_compute/graph/nodes/FusedDepthwiseConvolutionBatchNormalizationNode.h"
#include "arm_compute/graph/nodes/GenerateProposalsLayerNode.h"
#include "arm_compute/graph/nodes/InputNode.h"
#include "arm_compute/graph/nodes/L2NormalizeLayerNode.h"
#include "arm_compute/graph/nodes/NormalizationLayerNode.h"
#include "arm_compute/graph/nodes/NormalizePlanarYUVLayerNode.h"
#include "arm_compute/graph/nodes/OutputNode.h"
#include "arm_compute/graph/nodes/PReluLayerNode.h"
#include "arm_compute/graph/nodes/PadLayerNode.h"
#include "arm_compute/graph/nodes/PermuteLayerNode.h"
#include "arm_compute/graph/nodes/PoolingLayerNode.h"
#include "arm_compute/graph/nodes/PrintLayerNode.h"
#include "arm_compute/graph/nodes/PriorBoxLayerNode.h"
#include "arm_compute/graph/nodes/QuantizationLayerNode.h"
#include "arm_compute/graph/nodes/ROIAlignLayerNode.h"
#include "arm_compute/graph/nodes/ReductionLayerNode.h"
#include "arm_compute/graph/nodes/ReorgLayerNode.h"
#include "arm_compute/graph/nodes/ReshapeLayerNode.h"
#include "arm_compute/graph/nodes/ResizeLayerNode.h"
#include "arm_compute/graph/nodes/SliceLayerNode.h"
#include "arm_compute/graph/nodes/SoftmaxLayerNode.h"
#include "arm_compute/graph/nodes/SplitLayerNode.h"
#include "arm_compute/graph/nodes/StackLayerNode.h"
#include "arm_compute/graph/nodes/StridedSliceLayerNode.h"

#endif /* ARM_COMPUTE_GRAPH_NODES_H */
