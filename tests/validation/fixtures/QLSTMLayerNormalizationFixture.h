/*
 * Copyright (c) 2020 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_QLSTM_LAYER_NORMALIZATION_FIXTURE
#define ARM_COMPUTE_TEST_QLSTM_LAYER_NORMALIZATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/QLSTMLayerNormalization.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class QLSTMLayerNormalizationValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weight_shape, TensorShape bias_shape, DataType data_type, QuantizationInfo weight_qinfo)
    {
        ARM_COMPUTE_ERROR_ON(data_type != DataType::QSYMM16);

        _data_type = data_type;
        _qinfo     = weight_qinfo;

        _target    = compute_target(input_shape, weight_shape, bias_shape);
        _reference = compute_reference(input_shape, weight_shape, bias_shape);
    }

protected:
    template <typename InputType, typename BiasType>
    void fill(InputType &&input_tensor, InputType &&weight_tensor, BiasType &&bias_tensor)
    {
        switch(_data_type)
        {
            case DataType::QSYMM16:
            {
                // Value ranges are based on reference implementation's test case.
                constexpr int16_t input_min  = -1000;
                constexpr int16_t input_max  = 1000;
                constexpr int16_t weight_min = 19000;
                constexpr int16_t weight_max = 27000;
                constexpr int32_t bias_min   = -16000000;
                constexpr int32_t bias_max   = -13000000;

                std::uniform_int_distribution<> input_distribution(input_min, input_max);
                std::uniform_int_distribution<> weight_distribution(weight_min, weight_max);
                std::uniform_int_distribution<> bias_distribution(bias_min, bias_max);

                library->fill(input_tensor, input_distribution, 0);
                library->fill(weight_tensor, weight_distribution, 0);
                library->fill(bias_tensor, bias_distribution, 0);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("non-supported data type");
                break;
        }
    }

    void allocate_tensors(const std::vector<TensorType *> &tensors)
    {
        for(auto t : tensors)
        {
            ARM_COMPUTE_EXPECT(t->info()->is_resizable(), framework::LogLevel::ERRORS);
            t->allocator()->allocate();
            ARM_COMPUTE_EXPECT(!t->info()->is_resizable(), framework::LogLevel::ERRORS);
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weight_shape, const TensorShape &bias_shape)
    {
        TensorType input  = create_tensor<TensorType>(input_shape, _data_type, 1);
        TensorType weight = create_tensor<TensorType>(weight_shape, _data_type, 1, _qinfo);
        TensorType bias   = create_tensor<TensorType>(bias_shape, DataType::S32, 1);
        TensorType output = create_tensor<TensorType>(input_shape, _data_type, 1);

        FunctionType fn;
        fn.configure(&input, &output, &weight, &bias);
        allocate_tensors({ &input, &weight, &bias, &output });
        fill(AccessorType(input), AccessorType(weight), AccessorType(bias));

        ThreadInfo tinfo;
        tinfo.cpu_info = &NEScheduler::get().cpu_info();
        fn.run(fn.window(), tinfo);

        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weight_shape, const TensorShape &bias_shape)
    {
        // Create reference
        SimpleTensor<T>       input{ input_shape, _data_type, 1 };
        SimpleTensor<T>       weight{ weight_shape, _data_type, 1, _qinfo };
        SimpleTensor<int32_t> bias{ bias_shape, DataType::S32, 1 };

        // Fill reference
        fill(input, weight, bias);

        return reference::qlstm_layer_normalization(input, weight, bias);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    DataType         _data_type{};
    QuantizationInfo _qinfo{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* ARM_COMPUTE_TEST_QLSTM_LAYER_NORMALIZATION_FIXTURE */
