/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_ICLMATMULNATIVEKERNELVARIANT_H
#define ACL_SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_ICLMATMULNATIVEKERNELVARIANT_H

#include "arm_compute/core/CoreTypes.h" // DataType
#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/function_info/ActivationLayerInfo.h"
#include "arm_compute/function_info/MatMulInfo.h"

#include "src/core/common/Macros.h"

#include <array>

namespace arm_compute
{
namespace cl_matmul
{
enum class MatMulKernelType
{
    /** Native matrix multiplication for FP types */
    NATIVE_FP,

    /** Native matrix multiplication for quantized types */
    NATIVE_QUANTIZED,

    /** Native matrix multiplication using MMUL extension for FP types */
    NATIVE_MMUL_FP,

    /** Native matrix multiplication using MMUL extension for Quantized types */
    NATIVE_MMUL_QUANTIZED
};

/** Basic container for the OpenCL MatMul Native variant functions */
template <class T>
class ClMatMulNativeVariantArray
{
public:
    /** Alias for Float index */
    static constexpr size_t DT_FLOAT = 0;
    /** Alias for Quantized type index */
    static constexpr size_t DT_QUANTIZED = 1;

    /** Constructor
     *
     * @param[in] func_float     Function to call for matmul native float (F32, F16)
     * @param[in] func_quantized Function to call for matmul native quantized (QASYMM8, QASYMM8_SIGNED, QSYMM8_PER_CHANNEL)
     *
     */
    ClMatMulNativeVariantArray(T func_float, T func_quantized) : _configs{func_float, func_quantized}
    {
    }

    /** Method to return the matmul native variant function based on data type
     *
     * @param[in] data_type Input data type
     *
     * @return the valid function otherwise it returns nullptr if the data type is not valid
     */
    T get_function(DataType data_type)
    {
        switch (data_type)
        {
            case DataType::F32:
            case DataType::F16:
                return _configs.at(DT_FLOAT);
            case DataType::QASYMM8:
            case DataType::QASYMM8_SIGNED:
            case DataType::QSYMM8_PER_CHANNEL:
                return _configs.at(DT_QUANTIZED);
            default:
                return nullptr;
        }
    }

private:
    std::array<T, 2> _configs;
};

/** Basic interface for the matmul native kernel variant
 *  This is the base class that chooses architecture specific kernel variants.
*/
class IClMatMulNativeKernelVariant
{
public:
    /** Constructor
     *
     * @param[in] arch GPU target
     */
    IClMatMulNativeKernelVariant(GPUTarget arch) : _target(arch)
    {
    }
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(IClMatMulNativeKernelVariant);
    /** Virtual destructor */
    virtual ~IClMatMulNativeKernelVariant() = default;
    /** This method returns the @ref MatMulKernelType for the given inputs
     *
     * @param[in] lhs      LHS tensor
     * @param[in] rhs      RHS tensor
     * @param[in] info     MatMul info
     * @param[in] act_info Activation layer info
     */
    virtual MatMulKernelType select_kernel(const ITensorInfo         *lhs,
                                           const ITensorInfo         *rhs,
                                           const MatMulInfo          &info,
                                           const ActivationLayerInfo &act_info) = 0;

protected:
    GPUTarget _target;
};
} // namespace cl_matmul
} // namespace arm_compute
#endif // ACL_SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_ICLMATMULNATIVEKERNELVARIANT_H
