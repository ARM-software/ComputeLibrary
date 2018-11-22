/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_LSTMPARAMS_H__
#define __ARM_COMPUTE_LSTMPARAMS_H__

#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
template <typename T>
class LSTMParams
{
public:
    /** Constructor */
    LSTMParams()
        : _input_to_input_weights(nullptr), _recurrent_to_input_weights(nullptr), _cell_to_input_weights(nullptr), _input_gate_bias(nullptr), _cell_to_forget_weights(nullptr),
          _cell_to_output_weights(nullptr), _projection_weights(nullptr), _projection_bias(nullptr), _has_peephole_opt(false), _has_projection(false), _has_cifg_opt(true)
    {
    }
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    LSTMParams(const LSTMParams &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    LSTMParams &operator=(const LSTMParams &) = delete;
    /** Default destructor */
    ~LSTMParams() = default;
    /** Set CIFG tensor parameters.
     *
     * @param[in] input_to_input_weights     2D weights tensor with dimensions [input_size, num_units]. Data types supported: F16/F32.
     * @param[in] recurrent_to_input_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Same as @p input_to_input_weights.
     * @param[in] cell_to_input_weights      1D weights tensor with dimensions [num_units]. Can be nullptr. Data type supported: Same as @p input_to_input_weights.
     * @param[in] input_gate_bias            1D weights tensor with dimensions [num_units]. Data type supported: Same as @p input_to_input_weights
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_cifg_params(const T *input_to_input_weights, const T *recurrent_to_input_weights, const T *cell_to_input_weights, const T *input_gate_bias)
    {
        _input_to_input_weights     = input_to_input_weights;
        _recurrent_to_input_weights = recurrent_to_input_weights;
        _cell_to_input_weights      = cell_to_input_weights;
        _input_gate_bias            = input_gate_bias;
        _has_cifg_opt               = false;
        return *this;
    }
    /** Set projection tensor parameters.
     *
     * @param[in] projection_weights 2D weights tensor with dimensions [output_size, num_units]. Data type supported: Data types supported: F16/F32.
     * @param[in] projection_bias    1D weights tensor with dimensions [output_size]. Data type supported: Same as @p projection_weights.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_projection_params(const T *projection_weights, const T *projection_bias)
    {
        _projection_weights = projection_weights;
        _projection_bias    = projection_bias;
        _has_projection     = true;
        return *this;
    }
    /** Set peephole tensor parameters.
     *
     * @param[in] cell_to_forget_weights 1D weights tensor with dimensions [num_units]. Data type supported: Data types supported: F16/F32.
     * @param[in] cell_to_output_weights 1D weights tensor with dimensions [num_units]. Data type supported: Same as @p cell_to_input_weights.
     *
     * @return Reference to this LSTMParams object
     */
    LSTMParams &set_peephole_params(const T *cell_to_forget_weights, const T *cell_to_output_weights)
    {
        _cell_to_forget_weights = cell_to_forget_weights;
        _cell_to_output_weights = cell_to_output_weights;
        _has_peephole_opt       = true;
        return *this;
    }

    const T *input_to_input_weights() const
    {
        return _input_to_input_weights;
    }

    const T *recurrent_to_input_weights() const
    {
        return _recurrent_to_input_weights;
    }

    const T *cell_to_input_weights() const
    {
        return _cell_to_input_weights;
    }

    const T *input_gate_bias() const
    {
        return _input_gate_bias;
    }

    const T *cell_to_forget_weights() const
    {
        return _cell_to_forget_weights;
    }

    const T *cell_to_output_weights() const
    {
        return _cell_to_output_weights;
    }

    const T *projection_weights() const
    {
        return _projection_weights;
    }

    const T *projection_bias() const
    {
        return _projection_bias;
    }

    bool has_peephole_opt() const
    {
        return _has_peephole_opt;
    }

    bool has_projection() const
    {
        return _has_projection;
    }

    bool has_cifg_opt() const
    {
        return _has_cifg_opt;
    }

private:
    const T *_input_to_input_weights;
    const T *_recurrent_to_input_weights;
    const T *_cell_to_input_weights;
    const T *_input_gate_bias;
    const T *_cell_to_forget_weights;
    const T *_cell_to_output_weights;
    const T *_projection_weights;
    const T *_projection_bias;
    bool     _has_peephole_opt;
    bool     _has_projection;
    bool     _has_cifg_opt;
};
}
#endif /*__ARM_COMPUTE_LSTMPARAMS_H__ */
