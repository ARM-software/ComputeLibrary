/*
 * Copyright (c) 2018-2021, 2023-2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CASTFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CASTFIXTURE_H

#include "tests/validation/fixtures/DepthConvertLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class CastValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy)
    {
        if(std::is_same<TensorType, Tensor>::value &&  // Cpu
            (dt_in == DataType::F16 || dt_out == DataType::F16) && !CPUInfo::get().has_fp16())
        {
            return;
        }

        _target    = compute_target(shape, dt_in, dt_out, policy);
        _reference = compute_reference(shape, dt_in, dt_out, policy);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, DataType dt_in, DataType dt_out)
    {
        // Restricting range to avoid inf values
        if(dt_out == DataType::F16)
        {
            const int signed_min   = -32000;
            const int signed_max   = 32000;
            const int unsigned_min = 0;
            const int unsigned_max = 65000;

            switch(dt_in)
            {
                case DataType::U8:
                case DataType::QASYMM8:
                case DataType::QASYMM8_SIGNED:
                case DataType::QSYMM8:
                case DataType::QSYMM8_PER_CHANNEL:
                case DataType::QSYMM16:
                case DataType::QASYMM16:
                case DataType::S8:
                case DataType::F32:
                {
                    library->fill_tensor_uniform(tensor, i);
                    break;
                }
                case DataType::U16:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<uint16_t>(unsigned_min), static_cast<uint16_t>(unsigned_max));
                    break;
                }
                case DataType::S16:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<int16_t>(signed_min), static_cast<int16_t>(signed_max));
                    break;
                }
                case DataType::U32:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<uint32_t>(unsigned_min), static_cast<uint32_t>(unsigned_max));
                    break;
                }
                case DataType::S32:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<int32_t>(signed_min), static_cast<int32_t>(signed_max));
                    break;
                }
                case DataType::U64:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<uint64_t>(unsigned_min), static_cast<uint64_t>(unsigned_max));
                    break;
                }
                case DataType::S64:
                {
                    library->fill_tensor_uniform(tensor, i, static_cast<int64_t>(signed_min), static_cast<int64_t>(signed_max));
                    break;
                }
                default:
                    ARM_COMPUTE_ERROR("NOT SUPPORTED!");
            }
        }
        else
        {
            library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy)
    {
        // These are necessary but not used qinfo for creating tensor buffer for QSYMM8_PER_CHANNEL
        QuantizationInfo src_not_used_qinfo(0.25f, 2);
        QuantizationInfo dst_not_used_qinfo(0.5f, 2);

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, dt_in, 1, src_not_used_qinfo);
        TensorType dst = create_tensor<TensorType>(shape, dt_out, 1, dst_not_used_qinfo);

        // Create and configure function
        FunctionType cast;
        cast.configure(&src, &dst, policy);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0, dt_in, dt_out);

        // Compute function
        cast.run();

        return dst;
    }

    SimpleTensor<T2> compute_reference(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy)
    {
        // Create reference
        SimpleTensor<T1> src{ shape, dt_in, 1 };

        // Fill reference
        fill(src, 0, dt_in, dt_out);

        return reference::depth_convert<T1, T2>(src, dt_out, policy, 0);
    }

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CASTFIXTURE_H
