/*
 * Copyright (c) 2022 Arm Limited.
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
#ifndef SRC_RUNTIME_HEURISTICS_DWC_NATIVE_ICLDWCNATIVEKERNELCONFIG
#define SRC_RUNTIME_HEURISTICS_DWC_NATIVE_ICLDWCNATIVEKERNELCONFIG

#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/Types.h"
#include "src/core/common/Macros.h"

namespace arm_compute
{
namespace cl_dwc
{
/** Basic container for the OpenCL depthwise convolution configuration functions */
template <class T>
class ClDWCNativeConfigArray
{
public:
    /** Alias for F32 index */
    static constexpr size_t DT_F32 = 0;
    /** Alias for F16 index */
    static constexpr size_t DT_F16 = 1;
    /** Alias for Int8 index */
    static constexpr size_t DT_INT8 = 2;

    /** Constructor
     *
     * @param[in] func_f32  Function to call for depthwise convolution F32
     * @param[in] func_f16  Function to call for depthwise convolution F16
     * @param[in] func_int8 Function to call for depthwise convolution Int8 (QASYMM8, QASYMM8_SIGNED, QSYMM8_PER_CHANNEL)
     *
     */
    ClDWCNativeConfigArray(T func_f32, T func_f16, T func_int8)
        : _configs{ func_f32, func_f16, func_int8 }
    {
    }

    /** Method to return the depthwise convolution configuration function based on data type
     *
     * @param[in] data_type Input data type
     *
     * @return the valid function otherwise it returns nullptr if the data type is not valid
     */
    T get_function(DataType data_type)
    {
        switch(data_type)
        {
            case DataType::F32:
                return _configs.at(DT_F32);
            case DataType::F16:
                return _configs.at(DT_F16);
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
            case DataType::QSYMM8_PER_CHANNEL:
                return _configs.at(DT_INT8);
            default:
                return nullptr;
        }
    }

private:
    std::array<T, 3> _configs;
};

/** Basic interface for the depthwise convolution kernel configuration */
class IClDWCNativeKernelConfig
{
public:
    /** Constructor
     *
     * @param[in] arch GPU target
     */
    IClDWCNativeKernelConfig(GPUTarget arch)
        : _target(arch)
    {
    }
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(IClDWCNativeKernelConfig);
    /** Virtual destructor */
    virtual ~IClDWCNativeKernelConfig() = default;
    /** This method returns the @ref DWCComputeKernelInfo for the given inputs
     *
     * @param[in] src              Source tensor (activation tensor)
     * @param[in] wei              Weights tensor
     * @param[in] conv_info        Convolution info
     * @param[in] dilation         Kernel dilation
     * @param[in] depth_multiplier Output feature maps multiplier
     */
    virtual DWCComputeKernelInfo configure(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
                                           unsigned int depth_multiplier) = 0;

protected:
    GPUTarget _target;
};
} // namespace cl_dwc
} // namespace arm_compute
#endif /* SRC_RUNTIME_HEURISTICS_DWC_NATIVE_ICLDWCNATIVEKERNELCONFIG */
