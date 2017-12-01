/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_HISTOGRAM_FIXTURE
#define ARM_COMPUTE_TEST_HISTOGRAM_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Histogram.h"
#include "utils/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename DistributionType>
class HistogramValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        std::mt19937                            gen(library->seed());
        std::uniform_int_distribution<size_t>   distribution_size_t(1, 30);
        const size_t                            num_bins = distribution_size_t(gen);
        std::uniform_int_distribution<int32_t>  distribution_int32_t(0, 125);
        const size_t                            offset = distribution_int32_t(gen);
        std::uniform_int_distribution<uint32_t> distribution_uint32_t(1, 255 - offset);
        const size_t                            range = distribution_uint32_t(gen);

        _target    = compute_target(shape, data_type, num_bins, offset, range);
        _reference = compute_reference(shape, data_type, num_bins, offset, range);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, size_t num_bins, int32_t offset, uint32_t range)
    {
        // Create tensors
        TensorType       src = create_tensor<TensorType>(shape, data_type);
        TensorType       dst = create_tensor<TensorType>(TensorShape(num_bins), DataType::U32);
        DistributionType distribution_dst(num_bins, offset, range);

        // Create and configure function
        FunctionType histogram;
        histogram.configure(&src, &distribution_dst);
        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        histogram.run();

        // Copy the distribution in a tensor
        arm_compute::utils::map(distribution_dst, true);
        AccessorType accessor_dst = AccessorType(dst);
        uint32_t    *dst_data     = static_cast<uint32_t *>(accessor_dst.data());

        ARM_COMPUTE_EXPECT(accessor_dst.size() <= dst.info()->total_size(), framework::LogLevel::ERRORS);

        std::copy_n(distribution_dst.buffer(), num_bins, dst_data);
        arm_compute::utils::unmap(distribution_dst);
        return dst;
    }

    SimpleTensor<uint32_t> compute_reference(const TensorShape &shape, DataType data_type, size_t num_bins, int32_t offset, uint32_t range)
    {
        ARM_COMPUTE_ERROR_ON(data_type != DataType::U8);

        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src);

        // Compute reference
        return reference::histogram<T>(src, num_bins, offset, range);
    }

    TensorType             _target{};
    SimpleTensor<uint32_t> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_HISTOGRAM_FIXTURE */
