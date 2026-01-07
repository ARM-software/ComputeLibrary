/*
 * Copyright (c) 2026 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_TOPKVLAYERFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_TOPKVLAYERFIXTURE_H

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/validation/reference/TopKV.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <type_traits>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class TopKVValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape predictions_shape, uint32_t k, DataType input_data_type)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            input_data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }
        const TensorShape targets_shape(predictions_shape[1]);
        _data_type = input_data_type;
        _target    = compute_target(predictions_shape, targets_shape, k);
        _reference = compute_reference(predictions_shape, targets_shape, k);
    }

protected:
    // Fills prediction scores with small random noise, assigns a random target class to each sample,
    // and slightly boosts the target score so some targets fall inside the top-K and others do not.
    template <typename U, typename W>
    void fill(U &&predictions, W &&targets, int32_t i, uint32_t C)
    {
        std::mt19937 gen(0);
        // 1) Fill targets with valid class indices in [0, C-1]
        std::uniform_int_distribution<uint32_t> class_dist(0, C - 1);
        library->fill(targets, class_dist, i);
        const unsigned int N = targets.shape()[0];

        // 2) Float predictions (F32/F16): small noise + per-sample boost to target class
        if (_data_type == DataType::F32 || _data_type == DataType::F16)
        {
            std::normal_distribution<float> noise(0.f, 0.1f);
            library->fill(predictions, noise, i);

            // Mixture of boosts so output is not always 1
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            std::normal_distribution<float>       strong_boost(0.8f, 0.15f);
            std::normal_distribution<float>       medium_boost(0.35f, 0.15f);
            std::normal_distribution<float>       weak_boost(0.05f, 0.10f);

            for (unsigned int n = 0; n < N; ++n)
            {
                const uint32_t target = *reinterpret_cast<const uint32_t *>(targets(Coordinates{static_cast<int>(n)}));

                float       boost = 0.f;
                const float r     = u01(gen);
                if (r < 0.35f)
                {
                    boost = strong_boost(gen);
                }
                else if (r < 0.80f)
                {
                    boost = medium_boost(gen);
                }
                else
                {
                    boost = weak_boost(gen);
                }
                boost = std::max(-0.2f, std::min(boost, 1.5f));

                const Coordinates pc{static_cast<int>(target), static_cast<int>(n)};

                if (_data_type == DataType::F32)
                {
                    *reinterpret_cast<float *>(predictions(pc)) += boost;
                }
                else // F16 (half)
                {
                    auto *p = reinterpret_cast<half *>(predictions(pc));
                    *p      = static_cast<half>(static_cast<float>(*p) + boost);
                }
            }
            return;
        }

        // 3) Quantized predictions (QASYMM8 / QASYMM8_SIGNED): small integer noise + small boost
        if (_data_type == DataType::QASYMM8 || _data_type == DataType::QASYMM8_SIGNED)
        {
            const bool is_signed = (_data_type == DataType::QASYMM8_SIGNED);

            // Small integer noise
            if (is_signed)
            {
                std::uniform_int_distribution<int> noise(-6, 6);
                library->fill(predictions, noise, i);
            }
            else
            {
                std::uniform_int_distribution<int> noise(0, 12);
                library->fill(predictions, noise, i);
            }

            // Small, variable boost in quantized units
            std::uniform_int_distribution<int> boost_dist(0, 18);

            for (unsigned int n = 0; n < N; ++n)
            {
                const uint32_t target = *reinterpret_cast<const uint32_t *>(targets(Coordinates{static_cast<int>(n)}));

                const int         boost = boost_dist(gen);
                const Coordinates pc{static_cast<int>(target), static_cast<int>(n)};

                if (is_signed)
                {
                    auto     *p = reinterpret_cast<int8_t *>(predictions(pc));
                    const int v = static_cast<int>(*p) + boost;
                    *p          = static_cast<int8_t>(std::max(-128, std::min(127, v)));
                }
                else
                {
                    auto     *p = reinterpret_cast<uint8_t *>(predictions(pc));
                    const int v = static_cast<int>(*p) + boost;
                    *p          = static_cast<uint8_t>(std::max(0, std::min(255, v)));
                }
            }
            return;
        }

        // 4) S32 predictions: integer noise + per-sample integer boost to target class
        if (_data_type == DataType::S32)
        {
            // Small integer noise around 0
            std::uniform_int_distribution<int> noise(-20, 20);
            library->fill(predictions, noise, i);

            // Mixture of integer boosts so output is not always 1
            std::uniform_real_distribution<float> u01(0.f, 1.f);
            std::normal_distribution<float>       strong_boost(120.f, 25.f);
            std::normal_distribution<float>       medium_boost(45.f, 20.f);
            std::normal_distribution<float>       weak_boost(5.f, 10.f);

            for (unsigned int n = 0; n < N; ++n)
            {
                const uint32_t target = *reinterpret_cast<const uint32_t *>(targets(Coordinates{static_cast<int>(n)}));

                float       boost_f = 0.f;
                const float r       = u01(gen);
                if (r < 0.35f)
                {
                    boost_f = strong_boost(gen);
                }
                else if (r < 0.80f)
                {
                    boost_f = medium_boost(gen);
                }
                else
                {
                    boost_f = weak_boost(gen);
                }

                // Clamp boost and convert to integer units
                boost_f             = std::max(-50.f, std::min(boost_f, 200.f));
                const int32_t boost = static_cast<int32_t>(boost_f >= 0.f ? (boost_f + 0.5f) : (boost_f - 0.5f));

                const Coordinates pc{static_cast<int>(target), static_cast<int>(n)};

                auto *p = reinterpret_cast<int32_t *>(predictions(pc));
                // Saturating add (avoid UB on overflow)
                const int64_t v64     = static_cast<int64_t>(*p) + static_cast<int64_t>(boost);
                const int64_t clamped = std::max<int64_t>(std::numeric_limits<int32_t>::min(),
                                                          std::min<int64_t>(std::numeric_limits<int32_t>::max(), v64));
                *p                    = static_cast<int32_t>(clamped);
            }
            return;
        }
    }

    TensorType compute_target(const TensorShape &pred_shape, const TensorShape &targets_shape, uint32_t k)
    {
        // Create tensors
        TensorType pred    = create_tensor<TensorType>(pred_shape, _data_type, 1, QuantizationInfo());
        TensorType targets = create_tensor<TensorType>(targets_shape, DataType::U32, 1, QuantizationInfo());
        TensorType output;

        // Create and configure function
        FunctionType topkv_layer;
        topkv_layer.configure(&pred, &targets, &output, k);

        ARM_COMPUTE_ASSERT(pred.info()->is_resizable());
        ARM_COMPUTE_ASSERT(targets.info()->is_resizable());

        // Allocate tensors
        pred.allocator()->allocate();
        targets.allocator()->allocate();
        output.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!pred.info()->is_resizable());

        // Fill tensors
        const auto C{pred_shape[0]};
        fill(AccessorType(pred), AccessorType(targets), 0, C);

        topkv_layer.run();

        return output;
    }

    SimpleTensor<uint8_t> compute_reference(const TensorShape &pred_shape, const TensorShape &targets_shape, uint32_t k)
    {
        // Create reference
        SimpleTensor<T>        pred{pred_shape, _data_type, 1, QuantizationInfo()};
        SimpleTensor<uint32_t> targets{targets_shape, DataType::U32, 1, QuantizationInfo()};

        const auto C{pred_shape[0]};
        fill(pred, targets, 0, C);

        return reference::topkv(pred, targets, k);
    }

protected:
    TensorType            _target{};
    SimpleTensor<uint8_t> _reference{};
    DataType              _data_type{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_TOPKVLAYERFIXTURE_H
