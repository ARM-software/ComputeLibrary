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
#ifndef ARM_COMPUTE_TEST_GEMMLOWPOUTPUT_DATASET
#define ARM_COMPUTE_TEST_GEMMLOWPOUTPUT_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Utils.h"

using namespace arm_compute;

namespace arm_compute
{
namespace test
{
namespace datasets
{
class GEMMLowpFusedOffsetOutputDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, TensorShape, int32_t, int32_t, GEMMLowpOutputStageInfo>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator             a_it,
                 std::vector<TensorShape>::const_iterator             b_it,
                 std::vector<TensorShape>::const_iterator             c_it,
                 std::vector<int32_t>::const_iterator                 a_offset_it,
                 std::vector<int32_t>::const_iterator                 b_offset_it,
                 std::vector<GEMMLowpOutputStageInfo>::const_iterator output_stage_it)
            : _a_it{ std::move(a_it) },
              _b_it{ std::move(b_it) },
              _c_it{ std::move(c_it) },
              _a_offset_it{ std::move(a_offset_it) },
              _b_offset_it{ std::move(b_offset_it) },
              _output_stage_it{ std::move(output_stage_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "A=" << *_a_it << ":";
            description << "B=" << *_b_it << ":";
            description << "C=" << *_c_it << ":";
            description << "a_offset=" << *_a_offset_it << ":";
            description << "b_offset=" << *_b_offset_it << ":";
            description << "output_type=" << string_from_gemmlowp_output_stage((*_output_stage_it).type) << ":";
            description << "output_offset=" << (*_output_stage_it).gemmlowp_offset << ":";
            description << "output_multiplier=" << (*_output_stage_it).gemmlowp_multiplier << ":";
            description << "output_shift=" << (*_output_stage_it).gemmlowp_shift << ":";
            description << "output_min=" << (*_output_stage_it).gemmlowp_min_bound << ":";
            description << "output_max=" << (*_output_stage_it).gemmlowp_max_bound << ":";

            return description.str();
        }

        GEMMLowpFusedOffsetOutputDataset::type operator*() const
        {
            return std::make_tuple(*_a_it, *_b_it, *_c_it, *_a_offset_it, *_b_offset_it, *_output_stage_it);
        }

        iterator &operator++()
        {
            ++_a_it;
            ++_b_it;
            ++_c_it;
            ++_a_offset_it;
            ++_b_offset_it;
            ++_output_stage_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator             _a_it;
        std::vector<TensorShape>::const_iterator             _b_it;
        std::vector<TensorShape>::const_iterator             _c_it;
        std::vector<int32_t>::const_iterator                 _a_offset_it;
        std::vector<int32_t>::const_iterator                 _b_offset_it;
        std::vector<GEMMLowpOutputStageInfo>::const_iterator _output_stage_it;
    };

    iterator begin() const
    {
        return iterator(_a_shapes.begin(), _b_shapes.begin(), _c_shapes.begin(), _a_offset.begin(), _b_offset.begin(), _output_stage.begin());
    }

    int size() const
    {
        return std::min(_a_shapes.size(), std::min(_b_shapes.size(), std::min(_c_shapes.size(), std::min(_a_offset.size(), std::min(_b_offset.size(), _output_stage.size())))));
    }

    void add_config(TensorShape a, TensorShape b, TensorShape c, int32_t a_offset, int32_t b_offset, GEMMLowpOutputStageInfo output_stage)
    {
        _a_shapes.emplace_back(std::move(a));
        _b_shapes.emplace_back(std::move(b));
        _c_shapes.emplace_back(std::move(c));
        _a_offset.emplace_back(std::move(a_offset));
        _b_offset.emplace_back(std::move(b_offset));
        _output_stage.emplace_back(std::move(output_stage));
    }

    GEMMLowpOutputStageInfo OutputStageInfo(GEMMLowpOutputStageType type, int32_t offset, int32_t multiplier, int32_t shift, int32_t min, int32_t max)
    {
        GEMMLowpOutputStageInfo output_stage = GEMMLowpOutputStageInfo();
        output_stage.type                    = type;
        output_stage.gemmlowp_offset         = offset;
        output_stage.gemmlowp_multiplier     = multiplier;
        output_stage.gemmlowp_shift          = shift;
        output_stage.gemmlowp_min_bound      = min;
        output_stage.gemmlowp_max_bound      = max;
        return output_stage;
    }

protected:
    GEMMLowpFusedOffsetOutputDataset()                                    = default;
    GEMMLowpFusedOffsetOutputDataset(GEMMLowpFusedOffsetOutputDataset &&) = default;

private:
    std::vector<TensorShape>             _a_shapes{};
    std::vector<TensorShape>             _b_shapes{};
    std::vector<TensorShape>             _c_shapes{};
    std::vector<int32_t>                 _a_offset{};
    std::vector<int32_t>                 _b_offset{};
    std::vector<GEMMLowpOutputStageInfo> _output_stage{};
};

class SmallGEMMLowpFusedOffsetOutputDataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedOffsetOutputDataset()
    {
        add_config(TensorShape(21U, 1U), TensorShape(43U, 21U), TensorShape(43U, 1U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -200, 2, 13, 10, 210));
        add_config(TensorShape(21U, 13U), TensorShape(33U, 21U), TensorShape(33U, 13U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -100, 2, 13, 10, 210));
        add_config(TensorShape(31U, 3U), TensorShape(72U, 31U), TensorShape(72U, 3U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 13, 10, 210));
        add_config(TensorShape(52U, 13U), TensorShape(33U, 52U), TensorShape(33U, 13U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 100, 2, 13, 10, 210));
        add_config(TensorShape(52U, 26U), TensorShape(33U, 52U), TensorShape(33U, 26U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 13, 10, 210));
        add_config(TensorShape(31U, 27U), TensorShape(23U, 31U), TensorShape(23U, 27U), 18, 23, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 200, 2, 13, 10, 210));
        add_config(TensorShape(38U, 43U), TensorShape(21U, 38U), TensorShape(21U, 43U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -200, 2, 13, 10, 210));
        add_config(TensorShape(32U, 72U), TensorShape(17U, 32U), TensorShape(17U, 72U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -100, 2, 13, 10, 210));

        add_config(TensorShape(21U, 1U), TensorShape(43U, 21U), TensorShape(43U, 1U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601600, 10, 10, 210));
        add_config(TensorShape(21U, 13U), TensorShape(33U, 21U), TensorShape(33U, 13U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601600, 10, 10, 210));
        add_config(TensorShape(31U, 3U), TensorShape(72U, 31U), TensorShape(72U, 3U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 254601600, 10, 10, 210));
        add_config(TensorShape(52U, 26U), TensorShape(33U, 52U), TensorShape(33U, 26U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 1, 254601600, 10, 10, 210));
        add_config(TensorShape(31U, 27U), TensorShape(23U, 31U), TensorShape(23U, 27U), 5, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 2, 254601602, 10, 10, 210));
        add_config(TensorShape(38U, 43U), TensorShape(21U, 38U), TensorShape(21U, 43U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601602, 10, 10, 210));
        add_config(TensorShape(32U, 72U), TensorShape(17U, 32U), TensorShape(17U, 72U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601602, 10, 10, 210));
    }
};

class LargeGEMMLowpFusedOffsetOutputDataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    LargeGEMMLowpFusedOffsetOutputDataset()
    {
        add_config(TensorShape(923U, 1U), TensorShape(871U, 923U), TensorShape(871U, 1U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -200, 2, 18, 10, 210));
        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U, 429U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -100, 2, 18, 10, 210));
        add_config(TensorShape(873U, 7U), TensorShape(784U, 873U), TensorShape(784U, 7U), -1, 3, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 18, 10, 210));
        add_config(TensorShape(873U, 513U), TensorShape(784U, 873U), TensorShape(784U, 513U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 100, 2, 18, 10, 210));
        add_config(TensorShape(697U, 872U), TensorShape(563U, 697U), TensorShape(563U, 872U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 18, 10, 210));
        add_config(TensorShape(1021U, 973U), TensorShape(783U, 1021U), TensorShape(783U, 973U), 5, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 200, 2, 18, 10, 210));
        add_config(TensorShape(681U, 1023U), TensorShape(213U, 681U), TensorShape(213U, 1023U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -200, 2, 18, 10, 210));
        add_config(TensorShape(941U, 1011U), TensorShape(623U, 941U), TensorShape(623U, 1011U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -100, 2, 18, 10, 210));

        add_config(TensorShape(923U, 1U), TensorShape(871U, 923U), TensorShape(871U, 1U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601600, 15, 10, 210));
        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U, 429U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601600, 15, 10, 210));
        add_config(TensorShape(873U, 7U), TensorShape(784U, 873U), TensorShape(784U, 7U), -1, 3, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 254601600, 15, 10, 210));
        add_config(TensorShape(873U, 513U), TensorShape(784U, 873U), TensorShape(784U, 513U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 1, 254601600, 15, 10, 210));
        add_config(TensorShape(697U, 872U), TensorShape(563U, 697U), TensorShape(563U, 872U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 2, 254601602, 15, 10, 210));
        add_config(TensorShape(1021U, 973U), TensorShape(783U, 1021U), TensorShape(783U, 973U), 5, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601602, 15, 10, 210));
        add_config(TensorShape(681U, 1023U), TensorShape(213U, 681U), TensorShape(213U, 1023U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601602, 15, 10, 210));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMLOWPOUTPUT_DATASET */
