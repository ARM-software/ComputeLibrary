/*
 * Copyright (c) 2016-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CLDEPTHCONVERT_H
#define ARM_COMPUTE_CLDEPTHCONVERT_H

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/Types.h"

#include <memory>

namespace arm_compute
{
class CLCompileContext;
class ICLTensor;
class ITensorInfo;

/** Basic function to run @ref opencl::kernels::ClCastKernel */
class CLDepthConvertLayer : public IFunction
{
public:
    /** Constructor */
    CLDepthConvertLayer();
    /** Destructor */
    ~CLDepthConvertLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthConvertLayer(const CLDepthConvertLayer &) = delete;
    /** Default move constructor */
    CLDepthConvertLayer(CLDepthConvertLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLDepthConvertLayer &operator=(const CLDepthConvertLayer &) = delete;
    /** Default move assignment operator */
    CLDepthConvertLayer &operator=(CLDepthConvertLayer &&);
    /** Initialize the function's source, destination
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst                                   |
     * |:--------------|:-------------------------------------|
     * |U8             | S8, U16, S16, U32, S32, F16, F32     |
     * |U16            | U8, S8, S16, U32, S32, F16, F32      |
     * |S16            | U8, S8, U16, U32, S32, F16, F32      |
     * |U32            | U8, S8, U16, S16, S32, F16, F32      |
     * |S32            | U8, S8, U16, S16, U32, F16, F32      |
     * |F16            | U8, S8, U16, S16, U32, F32           |
     * |F32            | U8, S8, U16, S16, U32, F16           |
     *
     * Input data type must be different than output data type.
     *
     * @param[in]  input  The input tensor to convert. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[out] output The output tensor. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in]  policy Conversion policy.
     * @param[in]  shift  Value for down/up conversions. Must be 0 <= shift < 8.
     */
    void configure(const ICLTensor *input, ICLTensor *output, ConvertPolicy policy, uint32_t shift);
    /** Initialize the function's source, destination
     *
     * Input data type must be different than output data type.
     *
     * Valid conversions Input -> Output :
     *
     *   - U8  -> S8, U16, S16, U32, S32, F16, F32
     *   - U16 -> U8, S8, S16, U32, S32, F16, F32
     *   - S16 -> U8, S8, U16, U32, S32, F16, F32
     *   - U32 -> U8, S8, U16, S16, S32, F16, F32
     *   - S32 -> U8, S8, U16, S16, U32, F16, F32
     *   - F16 -> U8, S8, U16, S16, U32, F32
     *   - F32 -> U8, S8, U16, S16, U32, F16
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           The input tensor to convert. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[out] output          The output tensor. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in]  policy          Conversion policy.
     * @param[in]  shift           Value for down/up conversions. Must be 0 <= shift < 8.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, ICLTensor *output, ConvertPolicy policy, uint32_t shift);
    /** Static function to check if given info will lead to a valid configuration of @ref CLDepthConvertLayer
     *
     * @param[in] input  Source tensor info. Data types supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in] output Destination tensor info. Data type supported: U8/S8/U16/S16/U32/S32/F16/F32.
     * @param[in] policy Conversion policy.
     * @param[in] shift  Value for down/up conversions. Must be 0 <= shift < 8.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output, ConvertPolicy policy, uint32_t shift);

    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLDEPTHCONVERT_H*/
