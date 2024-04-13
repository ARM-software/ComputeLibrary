/*
 * Copyright (c) 2016-2024 Arm Limited.
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
#ifndef ACL_ARM_COMPUTE_FUNCTION_INFO_ACTIVATIONLAYERINFO_H
#define ACL_ARM_COMPUTE_FUNCTION_INFO_ACTIVATIONLAYERINFO_H

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/QuantizationInfo.h"

#include <array>
#include <memory>

#ifdef __aarch64__
#include <arm_neon.h>
#endif // __arch64__

namespace arm_compute
{
/** Available activation functions */
enum class ActivationFunction
{
    LOGISTIC,        /**< Logistic ( \f$ f(x) = \frac{1}{1 + e^{-x}} \f$ ) */
    TANH,            /**< Hyperbolic tangent ( \f$ f(x) = a \cdot tanh(b \cdot x) \f$ ) */
    RELU,            /**< Rectifier ( \f$ f(x) = max(0,x) \f$ ) */
    BOUNDED_RELU,    /**< Upper Bounded Rectifier ( \f$ f(x) = min(a, max(0,x)) \f$ ) */
    LU_BOUNDED_RELU, /**< Lower and Upper Bounded Rectifier ( \f$ f(x) = min(a, max(b,x)) \f$ ) */
    LEAKY_RELU, /**< Leaky Rectifier ( \f$ f(x) = \begin{cases}  \alpha x & \quad \text{if } x \text{ < 0}\\  x & \quad \text{if } x \geq \text{ 0 } \end{cases} \f$ ) */
    SOFT_RELU,  /**< Soft Rectifier ( \f$ f(x)= log(1+e^x) \f$ ) */
    ELU, /**< Exponential Linear Unit ( \f$ f(x) = \begin{cases}  \alpha (exp(x) - 1) & \quad \text{if } x \text{ < 0}\\  x & \quad \text{if } x \geq \text{ 0 } \end{cases} \f$ ) */
    ABS, /**< Absolute ( \f$ f(x)= |x| \f$ ) */
    SQUARE,     /**< Square ( \f$ f(x)= x^2 \f$ )*/
    SQRT,       /**< Square root ( \f$ f(x) = \sqrt{x} \f$ )*/
    LINEAR,     /**< Linear ( \f$ f(x)= ax + b \f$ ) */
    IDENTITY,   /**< Identity ( \f$ f(x)= x \f$ ) */
    HARD_SWISH, /**< Hard-swish ( \f$ f(x) = (x \text{ReLU6}(x+3))/6 = x \min(\max(0,x+3),6)/6 \f$ ) */
    SWISH,      /**< Swish ( \f$ f(x) = \frac{x}{1 + e^{-ax}} = x \text{logistic}(ax) \f$ ) */
    GELU        /**< GELU ( \f$ f(x) = x * 1/2 * 1 + erf(x / \sqrt{2}) \f$ ) */
};
/** Activation Layer Information class */
class ActivationLayerInfo
{
public:
    typedef arm_compute::ActivationFunction ActivationFunction;

    /** Lookup table  */
#ifdef __aarch64__
    using LookupTable256   = std::array<qasymm8_t, 256>;
    using LookupTable65536 = std::array<float16_t, 65536>;
#endif // __aarch64__

    ActivationLayerInfo() = default;
    /** Default Constructor
     *
     * @param[in] f The activation function to use.
     * @param[in] a (Optional) The alpha parameter used by some activation functions
     *              (@ref ActivationFunction::BOUNDED_RELU, @ref ActivationFunction::LU_BOUNDED_RELU, @ref ActivationFunction::LINEAR, @ref ActivationFunction::TANH).
     * @param[in] b (Optional) The beta parameter used by some activation functions (@ref ActivationFunction::LINEAR, @ref ActivationFunction::LU_BOUNDED_RELU, @ref ActivationFunction::TANH).
     */
    ActivationLayerInfo(ActivationFunction f, float a = 0.0f, float b = 0.0f) : _act(f), _a(a), _b(b), _enabled(true)
    {
    }
    /** Get the type of activation function */
    ActivationFunction activation() const
    {
        return _act;
    }
    /** Get the alpha value */
    float a() const
    {
        return _a;
    }
    /** Get the beta value */
    float b() const
    {
        return _b;
    }
    /** Check if initialised */
    bool enabled() const
    {
        return _enabled;
    }

#ifdef __aarch64__
    const LookupTable256 &lut() const
    {
        return _lut;
    }
    void setLookupTable256(LookupTable256 &lut)
    {
        _lut = std::move(lut);
    }

    const LookupTable65536 &lut_fp16() const
    {
        ARM_COMPUTE_ERROR_ON(_lut_fp16 == nullptr);
        return *_lut_fp16;
    }
    void setLookupTable65536(std::shared_ptr<LookupTable65536> lut)
    {
        _lut_fp16 = lut;
    }
#endif // __aarch64__
private:
    ActivationFunction _act     = {ActivationLayerInfo::ActivationFunction::IDENTITY};
    float              _a       = {};
    float              _b       = {};
    bool               _enabled = {false};

#ifdef __aarch64__
    LookupTable256                    _lut = {};
    std::shared_ptr<LookupTable65536> _lut_fp16{nullptr};
#endif // __aarch64__
};
} // namespace arm_compute
#endif // ACL_ARM_COMPUTE_FUNCTION_INFO_ACTIVATIONLAYERINFO_H
