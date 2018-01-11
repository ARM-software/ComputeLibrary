/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_DEPTHCONCATENATELAYERFIXTURE
#define ARM_COMPUTE_TEST_DEPTHCONCATENATELAYERFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace benchmark
{
/** Fixture that can be used for NE/CL/GC */
template <typename TensorType, typename ITensorType, typename Function, typename AccessorType>
class DepthConcatenateLayerFixture : public framework::Fixture
{
public:
    inline std::vector<TensorShape> generate_input_shapes(TensorShape shape)
    {
        // Create input shapes
        std::mt19937                    gen(library->seed());
        std::uniform_int_distribution<> num_dis(2, 6);
        const int                       num_tensors = num_dis(gen);

        std::vector<TensorShape>         shapes(num_tensors, shape);
        std::uniform_int_distribution<>  depth_dis(1, 7);
        std::bernoulli_distribution      mutate_dis(0.25f);
        std::uniform_real_distribution<> change_dis(-0.25f, 0.f);

        // Generate more shapes based on the input
        for(auto &s : shapes)
        {
            // Set the depth of the tensor
            s.set(2, depth_dis(gen));

            // Randomly change the first dimension
            if(mutate_dis(gen))
            {
                // Decrease the dimension by a small percentage. Don't increase
                // as that could make tensor too large. Also the change must be
                // an even number. Otherwise out depth concatenate fails.
                s.set(0, s[0] + 2 * static_cast<int>(s[0] * change_dis(gen)));
            }

            // Repeat the same as above for the second dimension
            if(mutate_dis(gen))
            {
                s.set(1, s[1] + 2 * static_cast<int>(s[1] * change_dis(gen)));
            }
        }

        return shapes;
    }

    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        // Generate input shapes
        std::vector<TensorShape> src_shapes = generate_input_shapes(shape);

        // Create tensors
        _srcs.reserve(src_shapes.size());

        std::vector<ITensorType *> src_ptrs;

        for(const auto &shape : src_shapes)
        {
            _srcs.emplace_back(create_tensor<TensorType>(shape, data_type, 1, _fractional_bits));
            src_ptrs.emplace_back(&_srcs.back());
        }

        TensorShape dst_shape = calculate_depth_concatenate_shape(src_ptrs);
        _dst                  = create_tensor<TensorType>(dst_shape, data_type, 1, _fractional_bits);

        _depth_concat.configure(src_ptrs, &_dst);

        for(auto &src : _srcs)
        {
            src.allocator()->allocate();
        }

        _dst.allocator()->allocate();
    }

    void run()
    {
        _depth_concat.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(_dst);
    }

    void teardown()
    {
        for(auto &src : _srcs)
        {
            src.allocator()->free();
        }

        _srcs.clear();

        _dst.allocator()->free();
    }

private:
    std::vector<TensorType> _srcs{};
    TensorType              _dst{};
    Function                _depth_concat{};
    int                     _fractional_bits{ 1 };
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHCONCATENATELAYERFIXTURE */
