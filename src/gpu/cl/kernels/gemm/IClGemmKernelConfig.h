/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_ICL_GEMM_KERNEL_CONFIG_H
#define ARM_COMPUTE_ICL_GEMM_KERNEL_CONFIG_H

#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Types.h"
#include "src/core/common/Macros.h"

#include <array>
namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace gemm
{
/** Basic container for the OpenCL GEMM configuration functions */
template <class T>
class CLGEMMConfigArray
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
     * @param[in] func_f32  Function to call for GEMM F32
     * @param[in] func_f16  Function to call for GEMM F16
     * @param[in] func_int8 Function to call for GEMM Int8 (QASYMM8, QASYMM8_SIGNED, QSYMM8_PER_CHANNEL)
     *
     */
    CLGEMMConfigArray(T func_f32, T func_f16, T func_int8)
        : _configs{ func_f32, func_f16, func_int8 }
    {
    }

    /** Method to return the GEMM configuration function based on data type
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

/** Basic interface for the GEMM kernel configuration */
class IClGemmKernelConfig
{
public:
    /** Constructor
     *
     * @param[in] arch GPU target
     */
    IClGemmKernelConfig(GPUTarget arch)
        : _target(arch)
    {
    }
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(IClGemmKernelConfig);
    /** Virtual destructor */
    virtual ~IClGemmKernelConfig() = default;
    /** Given M, N, K and B, this method returns the @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo to be used
     *
     * @param[in] m         Number of rows LHS matrix
     * @param[in] n         Number of columns RHS matrix
     * @param[in] k         Number of columns LHS matrix or number of rows RHS matrix
     * @param[in] b         Batch size
     * @param[in] data_type Data type
     */
    virtual std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type) = 0;

protected:
    GPUTarget _target;
};
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_ICL_GEMM_KERNEL_CONFIG_H */
