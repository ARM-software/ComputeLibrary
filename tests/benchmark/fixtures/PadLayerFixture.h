/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_PADLAYERFIXTURE
#define ARM_COMPUTE_TEST_PADLAYERFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
/** Fixture that can be used for NEON and CL */

template <typename TensorType, typename Accessor, typename Function, typename T>
class PaddingFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, const PaddingList &paddings, const PaddingMode mode)
    {
        PaddingList clamped_padding = paddings;
        if(mode != PaddingMode::CONSTANT)
        {
            // Clamp padding to prevent applying more than is possible.
            for(uint32_t i = 0; i < paddings.size(); ++i)
            {
                if(mode == PaddingMode::REFLECT)
                {
                    clamped_padding[i].first  = std::min(static_cast<uint64_t>(paddings[i].first), static_cast<uint64_t>(shape[i] - 1));
                    clamped_padding[i].second = std::min(static_cast<uint64_t>(paddings[i].second), static_cast<uint64_t>(shape[i] - 1));
                }
                else
                {
                    clamped_padding[i].first  = std::min(static_cast<uint64_t>(paddings[i].first), static_cast<uint64_t>(shape[i]));
                    clamped_padding[i].second = std::min(static_cast<uint64_t>(paddings[i].second), static_cast<uint64_t>(shape[i]));
                }
            }
        }

        const PixelValue const_value = PixelValue(static_cast<T>(0));

        TensorShape output_shape = arm_compute::misc::shape_calculator::compute_padded_shape(shape, paddings);

        // Create tensors
        src = create_tensor<TensorType>(shape, data_type);
        dst = create_tensor<TensorType>(output_shape, data_type);

        // Create and configure function
        pad_layer.configure(&src, &dst, paddings, const_value, mode);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        pad_layer.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        dst.allocator()->free();
    }

private:
    TensorType src{};
    TensorType dst{};
    Function   pad_layer{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PADLAYERFIXTURE */
