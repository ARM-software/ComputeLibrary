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
#ifndef ARM_COMPUTE_DETAIL_NEACTIVATION_FUNCTION_DETAIL_H
#define ARM_COMPUTE_DETAIL_NEACTIVATION_FUNCTION_DETAIL_H

#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace detail
{
/** Dummy activation object */
template <typename T, int S>
struct dummy
{
    /** Neon vector type. */
    using ExactType = typename wrapper::traits::neon_vector<T, S>::type;

    /** Construct a dummy activation object.
     *
     * @param[in] act_info Activation layer information.
     */
    explicit dummy(ActivationLayerInfo act_info)
    {
        ARM_COMPUTE_UNUSED(act_info);
    }

    /** Run activation function.
     *
     * @param[in] vval Vector of values.
     */
    void operator()(ExactType &vval)
    {
        ARM_COMPUTE_UNUSED(vval);
    }

    /** Run activation function.
     *
     * @param[in] val Scalar value.
     */
    void operator()(T &val)
    {
        ARM_COMPUTE_UNUSED(val);
    }
};
/** Linear activation object */
template <typename T, int S>
struct linear
{
    /** Neon vector type. */
    using ExactType = typename wrapper::traits::neon_vector<T, S>::type;
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    /** Construct a Linear activation object.
     *
     * @param[in] act_info Activation layer information.
     */
    explicit linear(ActivationLayerInfo act_info)
        : alpha(act_info.a()),
          beta(act_info.b()),
          valpha(wrapper::vdup_n(static_cast<T>(alpha), ExactTagType{})),
          vbeta(wrapper::vdup_n(static_cast<T>(beta), ExactTagType{}))
    {
    }

    /** Run activation function.
     *
     * @param[in] vval Vector of values.
     */
    void operator()(ExactType &vval)
    {
        vval = wrapper::vmla(vbeta, vval, valpha);
    }

    /** Run activation function.
     *
     * @param[in] val Scalar value.
     */
    void operator()(T &val)
    {
        val = alpha * val + beta;
    }

    const T         alpha;  /**< Scalar alpha */
    const T         beta;   /**< Scalar alpha */
    const ExactType valpha; /**< Vector of alphas. */
    const ExactType vbeta;  /**< Vector of betas. */
};
/** Square activation object */
template <typename T, int S>
struct square
{
    /** Neon vector type. */
    using ExactType = typename wrapper::traits::neon_vector<T, S>::type;
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    /** Construct a Square activation object.
     *
     * @param[in] act_info Activation layer information.
     */
    explicit square(ActivationLayerInfo act_info)
    {
        ARM_COMPUTE_UNUSED(act_info);
    }

    /** Run activation function.
     *
     * @param[in] vval Vector of values.
     */
    void operator()(ExactType &vval)
    {
        vval = wrapper::vmul(vval, vval);
    }

    /** Run activation function.
     *
     * @param[in] val Scalar value.
     */
    void operator()(T &val)
    {
        val = val * val;
    }
};
/** Logistic activation object */
template <typename T, int S>
struct logistic
{
    /** Neon vector type. */
    using ExactType = typename wrapper::traits::neon_vector<T, S>::type;
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    /** Construct a Logistic activation object.
     *
     * @param[in] act_info Activation layer information.
     */
    explicit logistic(ActivationLayerInfo act_info)
        : vone(wrapper::vdup_n(static_cast<T>(1), ExactTagType{}))
    {
        ARM_COMPUTE_UNUSED(act_info);
    }

    /** Run activation function.
     *
     * @param[in] vval Vector of values.
     */
    void operator()(ExactType &vval)
    {
        vval = wrapper::vinv(wrapper::vadd(vone, wrapper::vexpq(wrapper::vneg(vval))));
    }

    /** Run activation function.
     *
     * @param[in] val Scalar value.
     */
    void operator()(T &val)
    {
        val = 1 / (1 + std::exp(-val));
    }

    /** Vector of ones. */
    const ExactType vone;
};
/** RELU activation object */
template <typename T, int S>
struct relu
{
    /** Neon vector type. */
    using ExactType = typename wrapper::traits::neon_vector<T, S>::type;
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    /** Construct a RELU activation object.
     *
     * @param[in] act_info Activation layer information.
     */
    explicit relu(ActivationLayerInfo act_info)
        : vzero(wrapper::vdup_n(static_cast<T>(0), ExactTagType{}))
    {
        ARM_COMPUTE_UNUSED(act_info);
    }

    /** Run activation function.
     *
     * @param[in] vval Vector of values.
     */
    void operator()(ExactType &vval)
    {
        vval = wrapper::vmax(vzero, vval);
    }

    /** Run activation function.
     *
     * @param[in] val Scalar value.
     */
    void operator()(T &val)
    {
        val = std::max(static_cast<T>(0), val);
    }

    /** Vector of zeroes. */
    const ExactType vzero;
};
/** Bounded RELU activation object */
template <typename T, int S>
struct brelu
{
    /** Neon vector type. */
    using ExactType = typename wrapper::traits::neon_vector<T, S>::type;
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    /** Construct a bounded RELU activation object.
     *
     * @param[in] act_info Activation layer information.
     */
    explicit brelu(ActivationLayerInfo act_info)
        : alpha(act_info.a()),
          vzero(wrapper::vdup_n(static_cast<T>(0), ExactTagType{})),
          valpha(wrapper::vdup_n(static_cast<T>(act_info.a()), ExactTagType{}))
    {
    }

    /** Run activation function.
     *
     * @param[in] vval Vector of values.
     */
    void operator()(ExactType &vval)
    {
        vval = wrapper::vmin(valpha, wrapper::vmax(vzero, vval));
    }

    /** Run activation function.
     *
     * @param[in] val Scalar value.
     */
    void operator()(T &val)
    {
        val = std::min(alpha, std::max(static_cast<T>(0), val));
    }

    const T         alpha;  /** Scalar alpha */
    const ExactType vzero;  /** Vector of zeroes. */
    const ExactType valpha; /** Vector of alphas. */
};
/** Lower-Upper Bounded RELU activation object */
template <typename T, int S>
struct lubrelu
{
    /** Neon vector type. */
    using ExactType = typename wrapper::traits::neon_vector<T, S>::type;
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_vector<T, S>::tag_type;

    /** Construct a lower-upper bounded RELU activation object.
     *
     * @param[in] act_info Activation layer information.
     */
    explicit lubrelu(ActivationLayerInfo act_info)
        : alpha(act_info.a()),
          beta(act_info.b()),
          valpha(wrapper::vdup_n(static_cast<T>(act_info.a()), ExactTagType{})),
          vbeta(wrapper::vdup_n(static_cast<T>(act_info.b()), ExactTagType{}))
    {
    }

    /** Run activation function.
     *
     * @param[in] vval Vector of values.
     */
    void operator()(ExactType &vval)
    {
        vval = wrapper::vmin(valpha, wrapper::vmax(vbeta, vval));
    }

    /** Run activation function.
     *
     * @param[in] val Scalar value.
     */
    void operator()(T &val)
    {
        val = std::min(alpha, std::max(beta, val));
    }

    const T         alpha;  /**< Scalar alpha */
    const T         beta;   /**< Scalar alpha */
    const ExactType valpha; /** Vector of alphas. */
    const ExactType vbeta;  /** Vector of betas. */
};
} // namespace detail
} // namespace arm_compute
#endif /* ARM_COMPUTE_DETAIL_NEACTIVATION_FUNCTION_DETAIL_H */
