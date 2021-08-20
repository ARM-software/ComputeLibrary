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
#ifndef ARM_COMPUTE_ACL_DESCRIPTORS_H_
#define ARM_COMPUTE_ACL_DESCRIPTORS_H_

#ifdef __cplusplus
extern "C" {
#endif /** __cplusplus */

/**< Supported activation types */
typedef enum
{
    AclActivationTypeNone = 0,  /**< No activation */
    AclIdentity           = 1,  /**< Identity */
    AclLogistic           = 2,  /**< Logistic */
    AclTanh               = 3,  /**< Hyperbolic tangent */
    AclRelu               = 4,  /**< Rectifier */
    AclBoundedRelu        = 5,  /**< Upper Bounded Rectifier */
    AclLuBoundedRelu      = 6,  /**< Lower and Upper Bounded Rectifier */
    AclLeakyRelu          = 7,  /**< Leaky Rectifier */
    AclSoftRelu           = 8,  /**< Soft Rectifier */
    AclElu                = 9,  /**< Exponential Linear Unit */
    AclAbs                = 10, /**< Absolute */
    AclSquare             = 11, /**< Square */
    AclSqrt               = 12, /**< Square root */
    AclLinear             = 13, /**< Linear */
    AclHardSwish          = 14, /**< Hard-swish */
} AclActivationType;

/**< Activation layer descriptor */
typedef struct
{
    AclActivationType type;    /**< Activation type */
    float             a;       /**< Factor &alpha used by some activations */
    float             b;       /**< Factor &beta used by some activations */
    bool              inplace; /**< Hint that src and dst tensors will be the same */
} AclActivationDescriptor;
#ifdef __cplusplus
}
#endif /** __cplusplus */
#endif /* ARM_COMPUTE_ACL_DESCRIPTORS_H_ */
