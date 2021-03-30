/*
 * Copyright (c) 2020-2021 Arm Limited.
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

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/IWeightsManager.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "src/core/NEON/kernels/assembly/arm_gemm.hpp"
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

ARM_COMPUTE_TRACE_TO_STRING(ThresholdType)
using pair_uint = std::pair<unsigned int, unsigned int>;
ARM_COMPUTE_TRACE_TO_STRING(pair_uint)
ARM_COMPUTE_TRACE_TO_STRING(MemoryGroup)
ARM_COMPUTE_TRACE_TO_STRING(BoxNMSLimitInfo)
ARM_COMPUTE_TRACE_TO_STRING(DepthwiseConvolutionReshapeInfo)
ARM_COMPUTE_TRACE_TO_STRING(DWCWeightsKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(DWCKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(GEMMLHSMatrixInfo)
ARM_COMPUTE_TRACE_TO_STRING(GEMMRHSMatrixInfo)
ARM_COMPUTE_TRACE_TO_STRING(GEMMKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(InstanceNormalizationLayerKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(SoftmaxKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(FuseBatchNormalizationType)
ARM_COMPUTE_TRACE_TO_STRING(DirectConvolutionLayerOutputStageKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(FFTScaleKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(GEMMLowpOutputStageInfo)
ARM_COMPUTE_TRACE_TO_STRING(FFT1DInfo)
ARM_COMPUTE_TRACE_TO_STRING(FFT2DInfo)
ARM_COMPUTE_TRACE_TO_STRING(FFTDigitReverseKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(FFTRadixStageKernelInfo)
ARM_COMPUTE_TRACE_TO_STRING(IWeightsManager)
ARM_COMPUTE_TRACE_TO_STRING(Coordinates2D)
ARM_COMPUTE_TRACE_TO_STRING(ITensorInfo)
ARM_COMPUTE_TRACE_TO_STRING(InternalKeypoint)
ARM_COMPUTE_TRACE_TO_STRING(arm_gemm::Nothing)
ARM_COMPUTE_TRACE_TO_STRING(PixelValue)
ARM_COMPUTE_TRACE_TO_STRING(std::allocator<ITensor const *>)
using array_f32 = std::array<float, 9ul>;
ARM_COMPUTE_TRACE_TO_STRING(array_f32)

ARM_COMPUTE_CONST_REF_CLASS(arm_gemm::GemmArgs)
ARM_COMPUTE_CONST_REF_CLASS(arm_gemm::Nothing)
ARM_COMPUTE_CONST_REF_CLASS(arm_gemm::Activation)
ARM_COMPUTE_CONST_REF_CLASS(DirectConvolutionLayerOutputStageKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(GEMMLowpOutputStageInfo)
ARM_COMPUTE_CONST_REF_CLASS(DWCWeightsKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(DWCKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(DepthwiseConvolutionReshapeInfo)
ARM_COMPUTE_CONST_REF_CLASS(GEMMLHSMatrixInfo)
ARM_COMPUTE_CONST_REF_CLASS(GEMMRHSMatrixInfo)
ARM_COMPUTE_CONST_REF_CLASS(GEMMKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(InstanceNormalizationLayerKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(SoftmaxKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(PaddingMode)
ARM_COMPUTE_CONST_REF_CLASS(Coordinates)
ARM_COMPUTE_CONST_REF_CLASS(FFT1DInfo)
ARM_COMPUTE_CONST_REF_CLASS(FFT2DInfo)
ARM_COMPUTE_CONST_REF_CLASS(FFTDigitReverseKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(FFTRadixStageKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(FFTScaleKernelInfo)
ARM_COMPUTE_CONST_REF_CLASS(MemoryGroup)
ARM_COMPUTE_CONST_REF_CLASS(IWeightsManager)
ARM_COMPUTE_CONST_REF_CLASS(ActivationLayerInfo)
ARM_COMPUTE_CONST_REF_CLASS(PoolingLayerInfo)
ARM_COMPUTE_CONST_REF_CLASS(PadStrideInfo)
ARM_COMPUTE_CONST_REF_CLASS(NormalizationLayerInfo)
ARM_COMPUTE_CONST_REF_CLASS(Size2D)
ARM_COMPUTE_CONST_REF_CLASS(WeightsInfo)
ARM_COMPUTE_CONST_REF_CLASS(GEMMInfo)
ARM_COMPUTE_CONST_REF_CLASS(GEMMReshapeInfo)
ARM_COMPUTE_CONST_REF_CLASS(Window)
ARM_COMPUTE_CONST_REF_CLASS(BorderSize)
ARM_COMPUTE_CONST_REF_CLASS(BorderMode)
ARM_COMPUTE_CONST_REF_CLASS(PhaseType)
ARM_COMPUTE_CONST_REF_CLASS(MagnitudeType)
ARM_COMPUTE_CONST_REF_CLASS(Termination)
ARM_COMPUTE_CONST_REF_CLASS(ReductionOperation)
ARM_COMPUTE_CONST_REF_CLASS(InterpolationPolicy)
ARM_COMPUTE_CONST_REF_CLASS(SamplingPolicy)
ARM_COMPUTE_CONST_REF_CLASS(DataType)
ARM_COMPUTE_CONST_REF_CLASS(DataLayout)
ARM_COMPUTE_CONST_REF_CLASS(Channel)
ARM_COMPUTE_CONST_REF_CLASS(ConvertPolicy)
ARM_COMPUTE_CONST_REF_CLASS(TensorShape)
ARM_COMPUTE_CONST_REF_CLASS(PixelValue)
ARM_COMPUTE_CONST_REF_CLASS(Strides)
ARM_COMPUTE_CONST_REF_CLASS(WinogradInfo)
ARM_COMPUTE_CONST_REF_CLASS(RoundingPolicy)
ARM_COMPUTE_CONST_REF_CLASS(MatrixPattern)
ARM_COMPUTE_CONST_REF_CLASS(NonLinearFilterFunction)
ARM_COMPUTE_CONST_REF_CLASS(ThresholdType)
ARM_COMPUTE_CONST_REF_CLASS(ROIPoolingLayerInfo)
ARM_COMPUTE_CONST_REF_CLASS(BoundingBoxTransformInfo)
ARM_COMPUTE_CONST_REF_CLASS(ComparisonOperation)
ARM_COMPUTE_CONST_REF_CLASS(ArithmeticOperation)
ARM_COMPUTE_CONST_REF_CLASS(BoxNMSLimitInfo)
ARM_COMPUTE_CONST_REF_CLASS(FuseBatchNormalizationType)
ARM_COMPUTE_CONST_REF_CLASS(ElementWiseUnary)
ARM_COMPUTE_CONST_REF_CLASS(ComputeAnchorsInfo)
ARM_COMPUTE_CONST_REF_CLASS(PriorBoxLayerInfo)
ARM_COMPUTE_CONST_REF_CLASS(DetectionOutputLayerInfo)
ARM_COMPUTE_CONST_REF_CLASS(Coordinates2D)
ARM_COMPUTE_CONST_REF_CLASS(std::vector<const ITensor *>)
ARM_COMPUTE_CONST_REF_CLASS(std::vector<ITensor *>)
ARM_COMPUTE_CONST_REF_CLASS(std::vector<pair_uint>)
ARM_COMPUTE_CONST_REF_CLASS(pair_uint)
ARM_COMPUTE_CONST_REF_CLASS(array_f32)

ARM_COMPUTE_CONST_PTR_CLASS(ITensor)
ARM_COMPUTE_CONST_PTR_CLASS(ITensorInfo)
ARM_COMPUTE_CONST_PTR_CLASS(IWeightsManager)
ARM_COMPUTE_CONST_PTR_CLASS(InternalKeypoint)
ARM_COMPUTE_CONST_PTR_CLASS(Window)
ARM_COMPUTE_CONST_PTR_CLASS(HOGInfo)
ARM_COMPUTE_CONST_PTR_CLASS(std::allocator<ITensor const *>)
ARM_COMPUTE_CONST_PTR_CLASS(std::vector<unsigned int>)

ARM_COMPUTE_CONST_REF_SIMPLE(bool)
ARM_COMPUTE_CONST_REF_SIMPLE(uint64_t)
ARM_COMPUTE_CONST_REF_SIMPLE(int64_t)
ARM_COMPUTE_CONST_REF_SIMPLE(uint32_t)
ARM_COMPUTE_CONST_REF_SIMPLE(int32_t)
ARM_COMPUTE_CONST_REF_SIMPLE(int16_t)
ARM_COMPUTE_CONST_REF_SIMPLE(float)

ARM_COMPUTE_CONST_PTR_ADDRESS(float)
ARM_COMPUTE_CONST_PTR_ADDRESS(uint8_t)
ARM_COMPUTE_CONST_PTR_ADDRESS(void)
ARM_COMPUTE_CONST_PTR_ADDRESS(short)
ARM_COMPUTE_CONST_PTR_ADDRESS(int)
ARM_COMPUTE_CONST_PTR_ADDRESS(uint64_t)
ARM_COMPUTE_CONST_PTR_ADDRESS(uint32_t)
ARM_COMPUTE_CONST_PTR_ADDRESS(uint16_t)

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
