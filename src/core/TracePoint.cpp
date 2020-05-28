/*
 * Copyright (c) 2020 ARM Limited.
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
#include "arm_compute/core/TracePoint.h"

#include "arm_compute/core/HOGInfo.h"
#include "arm_compute/core/IArray.h"
#include "arm_compute/core/IDistribution1D.h"
#include "arm_compute/core/IHOG.h"
#include "arm_compute/core/ILut.h"
#include "arm_compute/core/IMultiHOG.h"
#include "arm_compute/core/IMultiImage.h"
#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/NEON/kernels/assembly/arm_gemm.hpp"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "utils/TypePrinter.h"

#include <array>
#include <cstdio>

namespace arm_compute
{
#ifndef DOXYGEN_SKIP_THIS
int TracePoint::g_depth = 0;

TracePoint::TracePoint(Layer layer, const std::string &class_name, void *object, Args &&args)
    : _depth(++g_depth)
{
    ARM_COMPUTE_UNUSED(layer, object, args);
    const std::string indentation = "  ";
    std::string       prefix      = "";
    for(int i = 0; i < _depth; ++i)
    {
        prefix += indentation;
        prefix += indentation;
    }
    printf("%s%s::configure(", prefix.c_str(), class_name.c_str());
    for(auto &arg : args.args)
    {
        printf("\n%s%s%s", prefix.c_str(), indentation.c_str(), arg.c_str());
    }
    printf("\n%s)\n", prefix.c_str());
}

TracePoint::~TracePoint()
{
    --g_depth;
}

std::string to_string(const arm_gemm::Activation &arg)
{
    switch(arg.type)
    {
        case arm_gemm::Activation::Type::None:
            return "None";
        case arm_gemm::Activation::Type::ReLU:
            return "ReLU";
        case arm_gemm::Activation::Type::BoundedReLU:
            return "BoundedReLU";
        default:
            ARM_COMPUTE_ERROR("Not supported");
            return "Uknown";
    };
}

std::string to_string(const arm_gemm::GemmArgs &arg)
{
    std::stringstream str;
    for(size_t k = 0; k < arg._ci->get_cpu_num(); ++k)
    {
        str << "[CPUCore " << k << "]" << to_string(arg._ci->get_cpu_model(0)) << " ";
    }
    str << "Msize= " << arg._Msize << " ";
    str << "Nsize= " << arg._Nsize << " ";
    str << "Ksize= " << arg._Ksize << " ";
    str << "nbatches= " << arg._nbatches << " ";
    str << "nmulti= " << arg._nmulti << " ";
    str << "trA= " << arg._trA << " ";
    str << "trB= " << arg._trB << " ";
    str << "Activation= " << to_string(arg._act) << " ";
    str << "maxthreads= " << arg._maxthreads << " ";
    str << "pretransposed_hint= " << arg._pretransposed_hint << " ";
    return str.str();
}

std::string to_string(const ITensor &arg)
{
    std::stringstream str;
    str << "TensorInfo(" << *arg.info() << ")";
    return str.str();
}

std::string to_ptr_string(const void *arg)
{
    std::stringstream ss;
    ss << arg;
    return ss.str();
}

TRACE_TO_STRING(ThresholdType)
TRACE_TO_STRING(IDetectionWindowArray)
TRACE_TO_STRING(ICoordinates2DArray)
TRACE_TO_STRING(IMultiImage)
using pair_uint = std::pair<unsigned int, unsigned int>;
TRACE_TO_STRING(pair_uint)
TRACE_TO_STRING(IKeyPointArray)
TRACE_TO_STRING(IDistribution1D)
TRACE_TO_STRING(IHOG)
TRACE_TO_STRING(ILut)
TRACE_TO_STRING(IPyramid)
TRACE_TO_STRING(IMultiHOG)
TRACE_TO_STRING(ISize2DArray)
TRACE_TO_STRING(MemoryGroup)
TRACE_TO_STRING(BoxNMSLimitInfo)
TRACE_TO_STRING(DepthwiseConvolutionReshapeInfo)
TRACE_TO_STRING(DWCWeightsKernelInfo)
TRACE_TO_STRING(DWCKernelInfo)
TRACE_TO_STRING(GEMMLHSMatrixInfo)
TRACE_TO_STRING(GEMMRHSMatrixInfo)
TRACE_TO_STRING(GEMMKernelInfo)
TRACE_TO_STRING(InstanceNormalizationLayerKernelInfo)
TRACE_TO_STRING(SoftmaxKernelInfo)
TRACE_TO_STRING(FuseBatchNormalizationType)
TRACE_TO_STRING(DirectConvolutionLayerOutputStageKernelInfo)
TRACE_TO_STRING(FFTScaleKernelInfo)
TRACE_TO_STRING(GEMMLowpOutputStageInfo)
TRACE_TO_STRING(FFT1DInfo)
TRACE_TO_STRING(FFT2DInfo)
TRACE_TO_STRING(FFTDigitReverseKernelInfo)
TRACE_TO_STRING(FFTRadixStageKernelInfo)
TRACE_TO_STRING(IWeightsManager)
TRACE_TO_STRING(Coordinates2D)
TRACE_TO_STRING(ITensorInfo)
TRACE_TO_STRING(InternalKeypoint)
TRACE_TO_STRING(arm_gemm::Nothing)
TRACE_TO_STRING(PixelValue)
TRACE_TO_STRING(std::allocator<ITensor const *>)
using array_f32 = std::array<float, 9ul>;
TRACE_TO_STRING(array_f32)

CONST_REF_CLASS(arm_gemm::GemmArgs)
CONST_REF_CLASS(arm_gemm::Nothing)
CONST_REF_CLASS(arm_gemm::Activation)
CONST_REF_CLASS(DirectConvolutionLayerOutputStageKernelInfo)
CONST_REF_CLASS(GEMMLowpOutputStageInfo)
CONST_REF_CLASS(DWCWeightsKernelInfo)
CONST_REF_CLASS(DWCKernelInfo)
CONST_REF_CLASS(DepthwiseConvolutionReshapeInfo)
CONST_REF_CLASS(GEMMLHSMatrixInfo)
CONST_REF_CLASS(GEMMRHSMatrixInfo)
CONST_REF_CLASS(GEMMKernelInfo)
CONST_REF_CLASS(InstanceNormalizationLayerKernelInfo)
CONST_REF_CLASS(SoftmaxKernelInfo)
CONST_REF_CLASS(PaddingMode)
CONST_REF_CLASS(Coordinates)
CONST_REF_CLASS(FFT1DInfo)
CONST_REF_CLASS(FFT2DInfo)
CONST_REF_CLASS(FFTDigitReverseKernelInfo)
CONST_REF_CLASS(FFTRadixStageKernelInfo)
CONST_REF_CLASS(FFTScaleKernelInfo)
CONST_REF_CLASS(MemoryGroup)
CONST_REF_CLASS(IWeightsManager)
CONST_REF_CLASS(ActivationLayerInfo)
CONST_REF_CLASS(PoolingLayerInfo)
CONST_REF_CLASS(PadStrideInfo)
CONST_REF_CLASS(NormalizationLayerInfo)
CONST_REF_CLASS(Size2D)
CONST_REF_CLASS(WeightsInfo)
CONST_REF_CLASS(GEMMInfo)
CONST_REF_CLASS(GEMMReshapeInfo)
CONST_REF_CLASS(Window)
CONST_REF_CLASS(BorderSize)
CONST_REF_CLASS(BorderMode)
CONST_REF_CLASS(PhaseType)
CONST_REF_CLASS(MagnitudeType)
CONST_REF_CLASS(Termination)
CONST_REF_CLASS(ReductionOperation)
CONST_REF_CLASS(InterpolationPolicy)
CONST_REF_CLASS(SamplingPolicy)
CONST_REF_CLASS(DataType)
CONST_REF_CLASS(DataLayout)
CONST_REF_CLASS(Channel)
CONST_REF_CLASS(ConvertPolicy)
CONST_REF_CLASS(TensorShape)
CONST_REF_CLASS(PixelValue)
CONST_REF_CLASS(Strides)
CONST_REF_CLASS(WinogradInfo)
CONST_REF_CLASS(RoundingPolicy)
CONST_REF_CLASS(MatrixPattern)
CONST_REF_CLASS(NonLinearFilterFunction)
CONST_REF_CLASS(ThresholdType)
CONST_REF_CLASS(ROIPoolingLayerInfo)
CONST_REF_CLASS(BoundingBoxTransformInfo)
CONST_REF_CLASS(ComparisonOperation)
CONST_REF_CLASS(ArithmeticOperation)
CONST_REF_CLASS(BoxNMSLimitInfo)
CONST_REF_CLASS(FuseBatchNormalizationType)
CONST_REF_CLASS(ElementWiseUnary)
CONST_REF_CLASS(ComputeAnchorsInfo)
CONST_REF_CLASS(PriorBoxLayerInfo)
CONST_REF_CLASS(DetectionOutputLayerInfo)
CONST_REF_CLASS(Coordinates2D)
CONST_REF_CLASS(std::vector<const ITensor *>)
CONST_REF_CLASS(std::vector<ITensor *>)
CONST_REF_CLASS(std::vector<pair_uint>)
CONST_REF_CLASS(pair_uint)
CONST_REF_CLASS(array_f32)

CONST_PTR_CLASS(ITensor)
CONST_PTR_CLASS(ITensorInfo)
CONST_PTR_CLASS(IWeightsManager)
CONST_PTR_CLASS(InternalKeypoint)
CONST_PTR_CLASS(IDetectionWindowArray)
CONST_PTR_CLASS(ICoordinates2DArray)
CONST_PTR_CLASS(IMultiImage)
CONST_PTR_CLASS(Window)
CONST_PTR_CLASS(IKeyPointArray)
CONST_PTR_CLASS(HOGInfo)
CONST_PTR_CLASS(IDistribution1D)
CONST_PTR_CLASS(IHOG)
CONST_PTR_CLASS(ILut)
CONST_PTR_CLASS(IPyramid)
CONST_PTR_CLASS(IMultiHOG)
CONST_PTR_CLASS(ISize2DArray)
CONST_PTR_CLASS(std::allocator<ITensor const *>)
CONST_PTR_CLASS(std::vector<unsigned int>)

CONST_REF_SIMPLE(bool)
CONST_REF_SIMPLE(uint64_t)
CONST_REF_SIMPLE(int64_t)
CONST_REF_SIMPLE(uint32_t)
CONST_REF_SIMPLE(int32_t)
CONST_REF_SIMPLE(int16_t)
CONST_REF_SIMPLE(float)

CONST_PTR_ADDRESS(float)
CONST_PTR_ADDRESS(uint8_t)
CONST_PTR_ADDRESS(void)
CONST_PTR_ADDRESS(short)
CONST_PTR_ADDRESS(int)
CONST_PTR_ADDRESS(uint64_t)
CONST_PTR_ADDRESS(uint32_t)
CONST_PTR_ADDRESS(uint16_t)

template <>
TracePoint::Args &&operator<<(TracePoint::Args &&tp, const uint16_t &arg)
{
    tp.args.push_back("uint16_t(" + support::cpp11::to_string<unsigned int>(arg) + ")");
    return std::move(tp);
}

template <>
TracePoint::Args &&operator<<(TracePoint::Args &&tp, const uint8_t &arg)
{
    tp.args.push_back("uint8_t(" + support::cpp11::to_string<unsigned int>(arg) + ")");
    return std::move(tp);
}
#endif /* DOXYGEN_SKIP_THIS */
} // namespace arm_compute
