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
#ifndef ARM_COMPUTE_NEELEMENTWISEUNARYLAYER_H
#define ARM_COMPUTE_NEELEMENTWISEUNARYLAYER_H

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
class ITensor;
class ITensorInfo;
/** Basic function to perform unary elementwise operations */
template <ElementWiseUnary op>
class NEElementwiseUnaryLayer : public IFunction
{
public:
    /** Default Constructor */
    NEElementwiseUnaryLayer();
    /** Default Destructor */
    ~NEElementwiseUnaryLayer();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseUnaryLayer(const NEElementwiseUnaryLayer &) = delete;
    /** Default move constructor */
    NEElementwiseUnaryLayer(NEElementwiseUnaryLayer &&);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEElementwiseUnaryLayer &operator=(const NEElementwiseUnaryLayer &) = delete;
    /** Default move assignment operator */
    NEElementwiseUnaryLayer &operator=(NEElementwiseUnaryLayer &&);

    /** Initialize the function
     *
     * Valid data layouts:
     * - All
     *
     * Valid data type configurations:
     * |src            |dst            |
     * |:--------------|:--------------|
     * |F16            |F16            |
     * |F32            |F32            |
     * |S32            |S32            |
     *
     * @param[in]  input  Input tensor. Data types supported: F16/F32, F16/F32/S32 for NEG/ABS operations.
     * @param[out] output Output tensor. Data types supported: Same as @p input.
     */
    void configure(const ITensor *input, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration
     *
     * @param[in] input  Input tensor info. Data types supported: F16/F32, F16/F32/S32 for NEG/ABS operations.
     * @param[in] output Output tensor info. Data types supported: Same as @p input.
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input, const ITensorInfo *output);
    // Inherited methods overridden:
    void run() override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

using NERsqrtLayer = NEElementwiseUnaryLayer<ElementWiseUnary::RSQRT>;
using NEExpLayer   = NEElementwiseUnaryLayer<ElementWiseUnary::EXP>;
using NENegLayer   = NEElementwiseUnaryLayer<ElementWiseUnary::NEG>;
using NELogLayer   = NEElementwiseUnaryLayer<ElementWiseUnary::LOG>;
using NEAbsLayer   = NEElementwiseUnaryLayer<ElementWiseUnary::ABS>;
using NERoundLayer = NEElementwiseUnaryLayer<ElementWiseUnary::ROUND>;
using NESinLayer   = NEElementwiseUnaryLayer<ElementWiseUnary::SIN>;

} // namespace arm_compute
#endif /* ARM_COMPUTE_NEELEMENTWISEUNARYLAYER_H */
