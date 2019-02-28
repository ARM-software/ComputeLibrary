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
#include "arm_compute/core/CPP/kernels/CPPTopKVKernel.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/Traits.h"

namespace arm_compute
{
namespace
{
template <typename T,
          typename std::enable_if<utils::traits::is_floating_point<T>::value, int>::type = 0>
inline bool greater_than(T a, T b)
{
    const T epsilon = std::numeric_limits<T>::epsilon();
    return (a - b > epsilon);
}

template < typename T,
           typename std::enable_if < !utils::traits::is_floating_point<T>::value, int >::type = 0 >
inline bool greater_than(T a, T b)
{
    return (a > b);
}

Status validate_arguments(const ITensorInfo *predictions, const ITensorInfo *targets, ITensorInfo *output, const unsigned int k)
{
    ARM_COMPUTE_UNUSED(k);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(predictions, 1, DataType::QASYMM8, DataType::S32, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(targets, 1, DataType::U32);

    ARM_COMPUTE_RETURN_ERROR_ON(predictions->num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(targets->num_dimensions() > 1);
    ARM_COMPUTE_RETURN_ERROR_ON(targets->dimension(0) != predictions->dimension(1));
    // Validate configured output
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), targets->tensor_shape());
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    }

    return Status{};
}
} // namespace

template <typename T>
void CPPTopKVKernel::run_topkv()
{
    for(unsigned int i = 0; i < _batch_size; ++i)
    {
        const auto target_class_id = *reinterpret_cast<uint32_t *>(_targets->ptr_to_element(Coordinates{ i }));
        const auto predicted_value = *reinterpret_cast<T *>(_predictions->ptr_to_element(Coordinates{ target_class_id, i }));

        // The variable rank indicates how many values there are before the target_class_id
        unsigned int rank = 0;
        for(unsigned int j = 0; (j < _num_classes) && (rank < _k); ++j)
        {
            const auto current_prediction = *reinterpret_cast<T *>(_predictions->ptr_to_element(Coordinates{ j, i }));
            if(greater_than(current_prediction, predicted_value))
            {
                rank++;
            }
        }
        *(_output->ptr_to_element(Coordinates{ i })) = static_cast<uint8_t>(rank < _k);
    }
}

CPPTopKVKernel::CPPTopKVKernel()
    : _predictions(nullptr), _targets(nullptr), _output(nullptr), _k(), _batch_size(), _num_classes()
{
}

void CPPTopKVKernel::configure(const ITensor *predictions, const ITensor *targets, ITensor *output, const unsigned int k)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(predictions, targets, output);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(predictions->info(), targets->info(), output->info(), k));
    auto_init_if_empty(*output->info(), targets->info()->tensor_shape(), 1, DataType::U8);

    _predictions = predictions;
    _targets     = targets;
    _output      = output;

    _k           = k;
    _batch_size  = predictions->info()->dimension(1);
    _num_classes = predictions->info()->dimension(0);

    ICPPKernel::configure(Window()); // Default 1 iteration window
}

Status CPPTopKVKernel::validate(const ITensorInfo *predictions, const ITensorInfo *targets, ITensorInfo *output, const unsigned int k)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(predictions, targets, output, k));
    return Status{};
}

bool CPPTopKVKernel::is_parallelisable() const
{
    return false;
}

void CPPTopKVKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(window, info);
    switch(_predictions->info()->data_type())
    {
        case DataType::F32:
            run_topkv<float>();
            break;
        case DataType::F16:
            run_topkv<half>();
            break;
        case DataType::S32:
            run_topkv<int>();
            break;
        case DataType::QASYMM8:
            run_topkv<uint8_t>();
            break;
        default:
            ARM_COMPUTE_ERROR("Not supported");
    }
}
} // namespace arm_compute
