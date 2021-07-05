/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef ARM_COMPUTE_ACL_OPERATORS_H_
#define ARM_COMPUTE_ACL_OPERATORS_H_

#include "arm_compute/AclDescriptors.h"
#include "arm_compute/AclTypes.h"

/** Used during an operator creation to validate its support */
#define ARM_COMPUTE_VALIDATE_OPERATOR_SUPPORT ((AclOperator *)(size_t)-1)

#ifdef __cplusplus
extern "C" {
#endif /** __cplusplus */

/** Create an activation operator
 *
 * Applies an activation function to a given tensor .
 * Compute Library supports a wide list of activation functions @ref AclActivationType.
 *
 * A summarized table is the following:
 *    Activation Function   |                                                    Mathematical Expression                                                               |
 * -------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
 *         Identity         |                                                       \f$ f(x)= x \f$                                                                    |
 *         Logistic         |                                               \f$ f(x) = \frac{1}{1 + e^{-x}} \f$                                                        |
 *           Tanh           |                                              \f$ f(x) = a \cdot tanh(b \cdot x) \f$                                                      |
 *           Relu           |                                                   \f$ f(x) = max(0,x) \f$                                                                |
 *       Bounded Relu       |                                                \f$ f(x) = min(a, max(0,x)) \f$                                                           |
 * Lower-Upper Bounded Relu |                                                \f$ f(x) = min(a, max(b,x)) \f$                                                           |
 *        Leaky Relu        |     \f$ f(x) = \begin{cases}  \alpha x & \quad \text{if } x \text{ < 0}\\  x & \quad \text{if } x \geq \text{ 0 } \end{cases} \f$        |
 *         Soft Relu        |                                                    \f$ f(x)= log(1+e^x) \f$                                                              |
 *         Soft Elu         | \f$ f(x) = \begin{cases}  \alpha (exp(x) - 1) & \quad \text{if } x \text{ < 0}\\  x & \quad \text{if } x \geq \text{ 0 } \end{cases} \f$ |
 *           Abs            |                                                      \f$ f(x)= |x| \f$                                                                   |
 *         Square           |                                                      \f$ f(x)= x^2 \f$                                                                   |
 *          Sqrt            |                                                   \f$ f(x) = \sqrt{x} \f$                                                                |
 *         Linear           |                                                    \f$ f(x)= ax + b \f$                                                                  |
 *       Hard Swish         |                                              \f$ f(x) = (x * relu6(x+3))/6 \f$                                                           |
 *
 * Backends:
 *   - OpenCL: ClActivationLayer
 *   - Cpu   : CpuActivationLayer
 *
 * @param[in, out] op   Operator construct to be created if creation was successful
 * @param[in]      ctx  Context to be used for the creation of the operator
 * @param[in]      src  Source tensor descriptor. Slot id: ACL_SRC
 * @param[in]      dst  Destination tensor descriptor. Slot id: ACL_DST
 * @param[in]      info Activation meta-data
 *
 * @return Status code
 *
 * Returns:
 *  - @ref AclSuccess if function was completed successfully
 *  - @ref AclOutOfMemory if there was a failure allocating memory resources
 *  - @ref AclUnsupportedTarget if operator for the requested target is unsupported
 *  - @ref AclInvalidArgument if a given argument is invalid
 */
AclStatus AclActivation(AclOperator                  *op,
                        AclContext                    ctx,
                        const AclTensorDescriptor    *src,
                        const AclTensorDescriptor    *dst,
                        const AclActivationDescriptor info);
#ifdef __cplusplus
}
#endif /** __cplusplus */
#endif /* ARM_COMPUTE_ACL_OPERATORS_H_ */
