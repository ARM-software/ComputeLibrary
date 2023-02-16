/*
 * Copyright (c) 2019-2022 Arm Limited.
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
            description << "output_multiplier={";
            for(auto it = (*_output_stage_it).gemmlowp_multipliers.begin(); it != (*_output_stage_it).gemmlowp_multipliers.end(); ++it)
            {
                description << (*it) << ", ";
            }
            description << "}:";
            description << "output_shift={";

            for(auto it = (*_output_stage_it).gemmlowp_shifts.begin(); it != (*_output_stage_it).gemmlowp_shifts.end(); ++it)
            {
                description << (*it) << ", ";
            }
            description << "}:";
            description << "output_min=" << (*_output_stage_it).gemmlowp_min_bound << ":";
            description << "output_max=" << (*_output_stage_it).gemmlowp_max_bound << ":";
            description << "is_quantized_per_channel=" << (*_output_stage_it).is_quantized_per_channel << ":";

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
        output_stage.gemmlowp_multipliers.push_back(multiplier);
        output_stage.gemmlowp_shifts.push_back(shift);
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

class SmallGEMMLowpFusedOffsetOutputUint8Dataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedOffsetOutputUint8Dataset()
    {
        add_config(TensorShape(21U, 13U), TensorShape(1U, 21U), TensorShape(1U, 13U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 13, 10, 210));
        add_config(TensorShape(52U, 13U), TensorShape(33U, 52U), TensorShape(33U, 13U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 100, 2, 13, 10, 210));
        add_config(TensorShape(31U, 27U), TensorShape(23U, 31U), TensorShape(23U, 27U), 18, 23, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 200, 2, 13, 10, 210));
        add_config(TensorShape(32U, 72U), TensorShape(16U, 32U), TensorShape(16U, 72U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 13, 10, 210));

        add_config(TensorShape(21U, 1U), TensorShape(43U, 21U), TensorShape(43U, 1U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601600, 10, 10, 210));
        add_config(TensorShape(31U, 3U), TensorShape(72U, 31U), TensorShape(72U, 3U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 254601600, 10, 10, 210));
        add_config(TensorShape(31U, 27U), TensorShape(23U, 31U), TensorShape(23U, 27U), 5, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 2, 254601602, 10, 10, 210));
        add_config(TensorShape(32U, 72U), TensorShape(17U, 32U), TensorShape(17U, 72U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601602, 10, 10, 210));
    }
};

class SmallGEMMLowpFusedBatchedMatMulDatasetUnsigned final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedBatchedMatMulDatasetUnsigned()
    {
        add_config(TensorShape(4U, 3U), TensorShape(2U, 4U), TensorShape(2U, 3U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 5, 1 << 25, 5, 0, 254));
        add_config(TensorShape(4U, 3U), TensorShape(2U, 4U), TensorShape(2U, 3U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 100, 1 << 25, 3, 0, 254));
        add_config(TensorShape(12U, 15U), TensorShape(7U, 12U), TensorShape(7U, 15U), -3, 15, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 1 << 19, 0, 20, 210));
        add_config(TensorShape(59U, 17U), TensorShape(36U, 59U), TensorShape(36U, 17U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -30, 2, 1 << 25, 14, 210));
        add_config(TensorShape(2U, 4U, 3U), TensorShape(5U, 2U, 3U), TensorShape(5U, 4U, 3U), -5, 12, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -20, 1 << 25, 4, 0, 127));
        add_config(TensorShape(15U, 7U, 3U), TensorShape(29U, 15U, 3U), TensorShape(29U, 7U, 3U), 5, 2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -10, 1 << 25, 6, 10, 210));
        add_config(TensorShape(56U, 17U, 32U), TensorShape(5U, 56U, 32U), TensorShape(5U, 17U, 32U), -3, 2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -15, 1 << 25, 3, 10, 210));
        add_config(TensorShape(13U, 256U, 32U), TensorShape(19U, 13U, 32U), TensorShape(19U, 256U, 32U), 5, 2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -15, 1 << 25, 6, 50, 225));
    }
};

class SmallGEMMLowpFusedBatchedMatMulDatasetSigned final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedBatchedMatMulDatasetSigned()
    {
        add_config(TensorShape(4U, 3U), TensorShape(2U, 4U), TensorShape(2U, 3U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 5, 1 << 25, 5, -128, 127));
        add_config(TensorShape(4U, 3U), TensorShape(2U, 4U), TensorShape(2U, 3U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 100, 1 << 25, 3, -128, 127));
        add_config(TensorShape(12U, 15U), TensorShape(7U, 12U), TensorShape(7U, 15U), -3, 15, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 1 << 19, 0, -108, 127));
        add_config(TensorShape(59U, 17U), TensorShape(36U, 59U), TensorShape(36U, 17U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -30, 2, 1 << 25, -98, 107));
        add_config(TensorShape(2U, 4U, 3U), TensorShape(5U, 2U, 3U), TensorShape(5U, 4U, 3U), -5, 12, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -20, 1 << 25, 4, -127, 64));
        add_config(TensorShape(15U, 7U, 3U), TensorShape(29U, 15U, 3U), TensorShape(29U, 7U, 3U), 5, 2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -10, 1 << 25, 6, -64, 127));
        add_config(TensorShape(56U, 17U, 32U), TensorShape(5U, 56U, 32U), TensorShape(5U, 17U, 32U), 3, 2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -15, 1 << 25, 6, -127, 110));
        add_config(TensorShape(13U, 256U, 32U), TensorShape(19U, 13U, 32U), TensorShape(19U, 256U, 32U), 5, 2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -15, 1 << 25, 6, -77, 115));
    }
};

class SmallGEMMLowpFusedOffsetOutputOutput3DUint8Dataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedOffsetOutputOutput3DUint8Dataset()
    {
        add_config(TensorShape(21U, 1421U, 33U), TensorShape(34U, 21U), TensorShape(34U, 7U, 203U, 33U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 13, 10, 210));
        add_config(TensorShape(31U, 102U, 55U), TensorShape(23U, 31U), TensorShape(23U, 1U, 102U, 55U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 100, 2, 13, 10, 210));
        add_config(TensorShape(38U, 1200U, 77U), TensorShape(21U, 38U), TensorShape(21U, 4U, 300U, 77U), 18, 23, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 200, 2, 13, 10, 210));
        add_config(TensorShape(32U, 103U, 99U), TensorShape(17U, 32U), TensorShape(17U, 1U, 103U, 99U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 13, 10, 210));
        add_config(TensorShape(16U, 1600U, 111U), TensorShape(8U, 16U), TensorShape(8U, 8U, 200U, 111U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601600, 10, 10,
                                                                                                                               210));
        add_config(TensorShape(16U, 1600U, 113U), TensorShape(8U, 16U), TensorShape(8U, 8U, 200U, 113U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 254601600, 10, 10,
                   210));
    }
};

class SmallGEMMLowpFusedOffsetOutputInputOutput3DUint8Dataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedOffsetOutputInputOutput3DUint8Dataset()
    {
        add_config(TensorShape(21U, 7U, 203U, 33U), TensorShape(34U, 21U), TensorShape(34U, 7U, 203U, 33U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 13, 10, 210));
        add_config(TensorShape(31U, 1U, 102U, 55U), TensorShape(23U, 31U), TensorShape(23U, 1U, 102U, 55U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 100, 2, 13, 10, 210));
        add_config(TensorShape(38U, 4U, 300U, 77U), TensorShape(21U, 38U), TensorShape(21U, 4U, 300U, 77U), 18, 23, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 200, 2, 13, 10, 210));
        add_config(TensorShape(32U, 1U, 103U, 99U), TensorShape(17U, 32U), TensorShape(17U, 1U, 103U, 99U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 13, 10, 210));
        add_config(TensorShape(16U, 8U, 200U, 111U), TensorShape(8U, 16U), TensorShape(8U, 8U, 200U, 111U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601600, 10, 10,
                   210));
        add_config(TensorShape(16U, 8U, 200U, 113U), TensorShape(8U, 16U), TensorShape(8U, 8U, 200U, 113U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 254601600, 10, 10,
                   210));
    }
};

class SmallGEMMLowpFusedOffsetOutputInt8Dataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedOffsetOutputInt8Dataset()
    {
        add_config(TensorShape(21U, 1U), TensorShape(1U, 21U), TensorShape(1U, 1U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -50, 2, 13, -10, 110));
        add_config(TensorShape(31U, 3U), TensorShape(72U, 31U), TensorShape(72U, 3U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 13, -10, 110));
        add_config(TensorShape(52U, 26U), TensorShape(33U, 52U), TensorShape(33U, 26U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 13, -10, 110));
        add_config(TensorShape(38U, 43U), TensorShape(21U, 38U), TensorShape(21U, 43U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -40, 2, 13, -10, 110));

        add_config(TensorShape(21U, 13U), TensorShape(33U, 21U), TensorShape(33U, 13U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601600, 10, -10, 110));
        add_config(TensorShape(52U, 26U), TensorShape(33U, 52U), TensorShape(33U, 26U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 1, 254601600, 10, -10, 110));
        add_config(TensorShape(38U, 43U), TensorShape(21U, 38U), TensorShape(21U, 43U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601602, 10, -10, 110));
        add_config(TensorShape(32U, 72U), TensorShape(17U, 32U), TensorShape(17U, 72U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601602, 10, -10, 110));
    }
};

class SmallGEMMLowpFusedOffsetOutputPerChannelDataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    SmallGEMMLowpFusedOffsetOutputPerChannelDataset()
    {
        add_config(TensorShape(21U, 1U, 6U), TensorShape(43U, 21U, 6U), TensorShape(43U, 1U, 6U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -200, 2, 13, 10, 210));
        add_config(TensorShape(21U, 13U, 3U), TensorShape(33U, 21U, 3U), TensorShape(33U, 13U, 3U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -100, 2, 13, 10, 210));
        add_config(TensorShape(31U, 3U, 2U), TensorShape(72U, 31U, 2U), TensorShape(72U, 3U, 2U), -2, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 13, 10, 210));
        add_config(TensorShape(52U, 13U, 7U), TensorShape(33U, 52U, 7U), TensorShape(33U, 13U, 7U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 100, 2, 13, 10, 210));
        add_config(TensorShape(52U, 26U, 8U), TensorShape(33U, 52U, 8U), TensorShape(33U, 26U, 8U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 13, 10, 210));
    }
};

class LargeGEMMLowpFusedOffsetOutputUint8Dataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    LargeGEMMLowpFusedOffsetOutputUint8Dataset()
    {
        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U, 429U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 18, 10, 210));
        add_config(TensorShape(873U, 513U), TensorShape(784U, 873U), TensorShape(784U, 513U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 100, 2, 18, 10, 210));
        add_config(TensorShape(1021U, 973U), TensorShape(783U, 1021U), TensorShape(783U, 973U), 5, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 200, 2, 18, 10, 210));
        add_config(TensorShape(941U, 1011U), TensorShape(623U, 941U), TensorShape(623U, 1011U), -9, 1, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -100, 2, 18, 10, 210));

        add_config(TensorShape(923U, 429U), TensorShape(871U, 923U), TensorShape(871U, 429U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601600, 15, 10, 210));
        add_config(TensorShape(873U, 513U), TensorShape(784U, 873U), TensorShape(784U, 513U), 0, 4, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 1, 254601600, 15, 10, 210));
        add_config(TensorShape(1021U, 973U), TensorShape(783U, 1021U), TensorShape(783U, 973U), 5, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601602, 15, 10, 210));
        add_config(TensorShape(681U, 1023U), TensorShape(213U, 681U), TensorShape(213U, 1023U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -1, 254601602, 15, 10, 210));
    }
};

class LargeGEMMLowpFusedOffsetOutputInt8Dataset final : public GEMMLowpFusedOffsetOutputDataset
{
public:
    LargeGEMMLowpFusedOffsetOutputInt8Dataset()
    {
        add_config(TensorShape(923U, 1U, 15U), TensorShape(871U, 923U, 15U), TensorShape(871U, 1U, 15U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -50, 2, 18, -10, 110));
        add_config(TensorShape(873U, 7U), TensorShape(784U, 873U), TensorShape(784U, 7U), -1, 3, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 18, -10, 110));
        add_config(TensorShape(697U, 872U), TensorShape(563U, 697U), TensorShape(563U, 872U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, 0, 2, 18, -10, 110));
        add_config(TensorShape(681U, 1023U), TensorShape(213U, 681U), TensorShape(213U, 1023U), -3, -2, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN, -50, 2, 18, -10, 110));

        add_config(TensorShape(923U, 1U), TensorShape(871U, 923U), TensorShape(871U, 1U), 0, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601600, 15, -10, 110));
        add_config(TensorShape(873U, 7U), TensorShape(784U, 873U), TensorShape(784U, 7U), -1, 3, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 0, 254601600, 15, -10, 110));
        add_config(TensorShape(697U, 872U), TensorShape(563U, 697U), TensorShape(563U, 872U), -2, 0, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, 2, 254601602, 15, -10, 110));
        add_config(TensorShape(1021U, 973U), TensorShape(783U, 1021U), TensorShape(783U, 973U), 5, 13, OutputStageInfo(GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT, -2, 254601602, 15, -10, 110));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMLOWPOUTPUT_DATASET */
