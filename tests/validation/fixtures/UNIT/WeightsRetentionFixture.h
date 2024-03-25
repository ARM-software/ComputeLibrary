/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_UNIT_WEIGHTS_RETENTION
#define ARM_COMPUTE_TEST_UNIT_WEIGHTS_RETENTION

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/FullyConnectedLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Test case to run a fully connected layer with weights retention, reconfigure
 *  with different shapes and rerun making sure the weights are retained.
 *
 * Runs a fully connected layer stimulating is_interleaved_transpose CLGEMM,
 * then reconfigures with different batch size and reruns.
 */
template <typename TensorType, typename AccessorType, typename FullyConnectedFunction>
class WeightsRetentionReconfigureTestCaseFixture : public framework::Fixture
{
    using T = float;

public:
    void setup()
    {
        _max_batches = 8;
        _cur_batches = 6;
        _target      = compute_target();
        _reference   = compute_reference();
    };

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(0.5f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target()
    {
        // Create tensors
        TensorType w1  = create_tensor<TensorType>(TensorShape(6000U, 15U), DataType::F32, 1);
        TensorType b1  = create_tensor<TensorType>(TensorShape(15U), DataType::F32, 1);
        TensorType src = create_tensor<TensorType>(TensorShape(1U, 15U, 400U, _max_batches), DataType::F32, 1);
        TensorType dst = create_tensor<TensorType>(TensorShape(15U, _max_batches), DataType::F32, 1);

        // Create and configure function
        FullyConnectedFunction fc_layer_1;
        fc_layer_1.configure(&src, &w1, &b1, &dst);

        // Allocate persistent tensors
        w1.allocator()->allocate();
        b1.allocator()->allocate();

        // Allocate tensors (1st iteration)
        src.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill tensors (1st iteration)
        fill(AccessorType(src), 0);
        fill(AccessorType(w1), 1);
        fill(AccessorType(b1), 2);

        // Compute functions (1st iteration)
        fc_layer_1.run();

        // Update tensor shapes (2nd iteration)
        auto src_padding     = src.allocator()->info().padding();
        auto dst_padding     = dst.allocator()->info().padding();
        int  diff            = _max_batches - _cur_batches;
        auto new_src_padding = PaddingSize(src_padding.top, src_padding.right, src_padding.bottom + diff, src_padding.left);
        auto new_dst_padding = PaddingSize(dst_padding.top, dst_padding.right, dst_padding.bottom + diff, dst_padding.left);
        src.allocator()->info().set_tensor_shape(TensorShape(1U, 15U, 400U, _cur_batches)).set_is_resizable(true).extend_padding(new_src_padding);
        src.allocator()->info().set_is_resizable(false);
        dst.allocator()->info().set_tensor_shape(TensorShape(15U, _cur_batches)).set_is_resizable(true).extend_padding(new_dst_padding);
        dst.allocator()->info().set_is_resizable(false);

        // Configure FC info
        FullyConnectedLayerInfo fc_info;
        fc_info.retain_internal_weights = true;

        // Configure functions (2nd iteration)
        fc_layer_1.configure(&src, &w1, &b1, &dst, fc_info);

        // Fill tensors (2nd iteration)
        fill(AccessorType(src), 5);

        // Compute functions (2nd iteration)
        fc_layer_1.run();

        return dst;
    }

    SimpleTensor<T> compute_reference()
    {
        // Create reference
        SimpleTensor<T> w1{ TensorShape(6000U, 15U), DataType::F32 };
        SimpleTensor<T> b1{ TensorShape(15U), DataType::F32 };
        SimpleTensor<T> src{ TensorShape(1U, 15U, 400U, _cur_batches), DataType::F32 };

        // Fill reference
        fill(src, 5);
        fill(w1, 1);
        fill(b1, 2);

        return reference::fully_connected_layer(src, w1, b1, TensorShape(15U, _cur_batches));
    }

protected:
    TensorType      _target{};
    SimpleTensor<T> _reference{};
    unsigned int    _max_batches{};
    unsigned int    _cur_batches{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_WEIGHTS_RETENTION */
