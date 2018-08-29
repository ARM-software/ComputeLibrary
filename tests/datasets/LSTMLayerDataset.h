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
#ifndef ARM_COMPUTE_TEST_LSTM_LAYER_DATASET
#define ARM_COMPUTE_TEST_LSTM_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class LSTMLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, TensorShape, TensorShape, TensorShape, TensorShape, ActivationLayerInfo, float, float>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator         src_it,
                 std::vector<TensorShape>::const_iterator         input_weights_it,
                 std::vector<TensorShape>::const_iterator         recurrent_weights_it,
                 std::vector<TensorShape>::const_iterator         cells_bias_it,
                 std::vector<TensorShape>::const_iterator         output_cell_it,
                 std::vector<TensorShape>::const_iterator         dst_it,
                 std::vector<TensorShape>::const_iterator         scratch_it,
                 std::vector<ActivationLayerInfo>::const_iterator infos_it,
                 std::vector<float>::const_iterator               cell_threshold_it,
                 std::vector<float>::const_iterator               projection_threshold_it)
            : _src_it{ std::move(src_it) },
              _input_weights_it{ std::move(input_weights_it) },
              _recurrent_weights_it{ std::move(recurrent_weights_it) },
              _cells_bias_it{ std::move(cells_bias_it) },
              _output_cell_it{ std::move(output_cell_it) },
              _dst_it{ std::move(dst_it) },
              _scratch_it{ std::move(scratch_it) },
              _infos_it{ std::move(infos_it) },
              _cell_threshold_it{ std::move(cell_threshold_it) },
              _projection_threshold_it{ std::move(projection_threshold_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "InputWeights=" << *_input_weights_it << ":";
            description << "RecurrentWeights=" << *_recurrent_weights_it << ":";
            description << "Biases=" << *_cells_bias_it << ":";
            description << "Scratch=" << *_scratch_it << ":";
            description << "Out=" << *_dst_it;
            return description.str();
        }

        LSTMLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_input_weights_it, *_recurrent_weights_it, *_cells_bias_it, *_output_cell_it, *_dst_it, *_scratch_it, *_infos_it, *_cell_threshold_it, *_projection_threshold_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_input_weights_it;
            ++_recurrent_weights_it;
            ++_cells_bias_it;
            ++_output_cell_it;
            ++_dst_it;
            ++_scratch_it;
            ++_infos_it;
            ++_cell_threshold_it;
            ++_projection_threshold_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator         _src_it;
        std::vector<TensorShape>::const_iterator         _input_weights_it;
        std::vector<TensorShape>::const_iterator         _recurrent_weights_it;
        std::vector<TensorShape>::const_iterator         _cells_bias_it;
        std::vector<TensorShape>::const_iterator         _output_cell_it;
        std::vector<TensorShape>::const_iterator         _dst_it;
        std::vector<TensorShape>::const_iterator         _scratch_it;
        std::vector<ActivationLayerInfo>::const_iterator _infos_it;
        std::vector<float>::const_iterator               _cell_threshold_it;
        std::vector<float>::const_iterator               _projection_threshold_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _input_weights_shapes.begin(), _recurrent_weights_shapes.begin(), _cell_bias_shapes.begin(), _output_cell_shapes.begin(), _dst_shapes.begin(),
                        _scratch_shapes.begin(), _infos.begin(), _cell_threshold.begin(), _projection_threshold.begin());
    }

    int size() const
    {
        return std::min(_src_shapes.size(), std::min(_input_weights_shapes.size(), std::min(_recurrent_weights_shapes.size(), std::min(_cell_bias_shapes.size(), std::min(_output_cell_shapes.size(),
                                                                                            std::min(_dst_shapes.size(), std::min(_scratch_shapes.size(), std::min(_cell_threshold.size(), std::min(_projection_threshold.size(), _infos.size())))))))));
    }

    void add_config(TensorShape src, TensorShape input_weights, TensorShape recurrent_weights, TensorShape cell_bias_weights, TensorShape output_cell_state, TensorShape dst, TensorShape scratch,
                    ActivationLayerInfo info, float cell_threshold, float projection_threshold)
    {
        _src_shapes.emplace_back(std::move(src));
        _input_weights_shapes.emplace_back(std::move(input_weights));
        _recurrent_weights_shapes.emplace_back(std::move(recurrent_weights));
        _cell_bias_shapes.emplace_back(std::move(cell_bias_weights));
        _output_cell_shapes.emplace_back(std::move(output_cell_state));
        _dst_shapes.emplace_back(std::move(dst));
        _scratch_shapes.emplace_back(std::move(scratch));
        _infos.emplace_back(std::move(info));
        _cell_threshold.emplace_back(std::move(cell_threshold));
        _projection_threshold.emplace_back(std::move(projection_threshold));
    }

protected:
    LSTMLayerDataset()                    = default;
    LSTMLayerDataset(LSTMLayerDataset &&) = default;

private:
    std::vector<TensorShape>         _src_shapes{};
    std::vector<TensorShape>         _input_weights_shapes{};
    std::vector<TensorShape>         _recurrent_weights_shapes{};
    std::vector<TensorShape>         _cell_bias_shapes{};
    std::vector<TensorShape>         _output_cell_shapes{};
    std::vector<TensorShape>         _dst_shapes{};
    std::vector<TensorShape>         _scratch_shapes{};
    std::vector<ActivationLayerInfo> _infos{};
    std::vector<float>               _cell_threshold{};
    std::vector<float>               _projection_threshold{};
};

class SmallLSTMLayerDataset final : public LSTMLayerDataset
{
public:
    SmallLSTMLayerDataset()
    {
        add_config(TensorShape(8U), TensorShape(8U, 16U), TensorShape(16U, 16U), TensorShape(16U), TensorShape(16U), TensorShape(16U), TensorShape(64U),
                   ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU), 0.05f, 0.93f);
        add_config(TensorShape(8U, 2U), TensorShape(8U, 16U), TensorShape(16U, 16U), TensorShape(16U), TensorShape(16U, 2U), TensorShape(16U, 2U), TensorShape(64U, 2U),
                   ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU), 0.05f, 0.93f);
        add_config(TensorShape(8U, 2U), TensorShape(8U, 16U), TensorShape(16U, 16U), TensorShape(16U), TensorShape(16U, 2U), TensorShape(16U, 2U), TensorShape(48U, 2U),
                   ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU), 0.05f, 0.93f);
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LSTM_LAYER_DATASET */
