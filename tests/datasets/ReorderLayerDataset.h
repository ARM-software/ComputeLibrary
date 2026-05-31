/*
 * Copyright (c) 2023, 2025 Arm Limited.
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
#ifndef ACL_TESTS_DATASETS_REORDERLAYERDATASET_H
#define ACL_TESTS_DATASETS_REORDERLAYERDATASET_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "utils/TypePrinter.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** [ReorderLayer datasets] **/
class ReorderLayerDataset
{
public:
    using type = std::tuple<TensorShape, TensorShape, WeightFormat>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator  in_it,
                 std::vector<TensorShape>::const_iterator  out_it,
                 std::vector<WeightFormat>::const_iterator _wf_in_it)
            : _in_it{std::move(in_it)}, _out_it{std::move(out_it)}, _wf_in_it{std::move(_wf_in_it)}
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_in_it << ":";
            description << "Out=" << *_out_it << ":";
            description << "Wf_In=" << *_wf_in_it << ":";
            return description.str();
        }

        ReorderLayerDataset::type operator*() const
        {
            return std::make_tuple(*_in_it, *_out_it, *_wf_in_it);
        }

        iterator &operator++()
        {
            ++_in_it;
            ++_out_it;
            ++_wf_in_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator  _in_it;
        std::vector<TensorShape>::const_iterator  _out_it;
        std::vector<WeightFormat>::const_iterator _wf_in_it;
    };

    iterator begin() const
    {
        return iterator(_in_shapes.begin(), _out_shapes.begin(), _in_wfs.begin());
    }

    int size() const
    {
        return std::min(_in_shapes.size(), std::min(_out_shapes.size(), _in_wfs.size()));
    }

    void add_config(TensorShape in, TensorShape out, WeightFormat in_wf)
    {
        _in_shapes.emplace_back(std::move(in));
        _out_shapes.emplace_back(std::move(out));
        _in_wfs.emplace_back(std::move(in_wf));
    }

    // protected:
    ReorderLayerDataset()                       = default;
    ReorderLayerDataset(ReorderLayerDataset &&) = default;

private:
    std::vector<TensorShape>  _in_shapes{};
    std::vector<TensorShape>  _out_shapes{};
    std::vector<WeightFormat> _in_wfs{};
};

/** [ReorderLayer datasets] **/

class ReorderLayerDatasetInterleave4 final : public ReorderLayerDataset
{
public:
    ReorderLayerDatasetInterleave4()
    {
        add_config(TensorShape(10U, 9U), TensorShape(10U, 12U), WeightFormat::OHWI);
        add_config(TensorShape(16U, 16U), TensorShape(16U, 16U), WeightFormat::OHWI);
        add_config(TensorShape(10U, 511U), TensorShape(10U, 512U), WeightFormat::OHWI);
        add_config(TensorShape(234U, 301U), TensorShape(234U, 304U), WeightFormat::OHWI);
        add_config(TensorShape(1024U, 1024U), TensorShape(1024U, 1024U), WeightFormat::OHWI);
        add_config(TensorShape(10U, 9U, 1U, 1U), TensorShape(10U, 12U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(16U, 16U, 1U, 1U), TensorShape(16U, 16U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(10U, 511U, 1U, 1U), TensorShape(10U, 512U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(234U, 301U, 1U, 1U), TensorShape(234U, 304U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(1024U, 1024U, 1U, 1U), TensorShape(1024U, 1024U, 1U, 1U), WeightFormat::OHWI);
    }
};

class ReorderLayerDatasetInterleave8 final : public ReorderLayerDataset
{
public:
    ReorderLayerDatasetInterleave8()
    {
        add_config(TensorShape(10U, 9U), TensorShape(10U, 16U), WeightFormat::OHWI);
        add_config(TensorShape(16U, 16U), TensorShape(16U, 16U), WeightFormat::OHWI);
        add_config(TensorShape(10U, 511U), TensorShape(10U, 512U), WeightFormat::OHWI);
        add_config(TensorShape(234U, 301U), TensorShape(234U, 304U), WeightFormat::OHWI);
        add_config(TensorShape(1024U, 1024U), TensorShape(1024U, 1024U), WeightFormat::OHWI);
        add_config(TensorShape(10U, 9U, 1U, 1U), TensorShape(10U, 16U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(16U, 16U, 1U, 1U), TensorShape(16U, 16U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(10U, 511U, 1U, 1U), TensorShape(10U, 512U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(234U, 301U, 1U, 1U), TensorShape(234U, 304U, 1U, 1U), WeightFormat::OHWI);
        add_config(TensorShape(1024U, 1024U, 1U, 1U), TensorShape(1024U, 1024U, 1U, 1U), WeightFormat::OHWI);
    }
};

class ReorderLayerDatasetInterleave4Block4 final : public ReorderLayerDataset
{
public:
    ReorderLayerDatasetInterleave4Block4()
    {
        add_config(TensorShape(12U, 9U), TensorShape(12U, 12U), WeightFormat::OHWI);
        add_config(TensorShape(16U, 16U), TensorShape(16U, 16U), WeightFormat::OHWI);
        add_config(TensorShape(12U, 511U), TensorShape(12U, 512U), WeightFormat::OHWI);
        add_config(TensorShape(244U, 301U), TensorShape(244U, 304U), WeightFormat::OHWI);
    }
};

class ReorderLayerDatasetInterleave8Block4 final : public ReorderLayerDataset
{
public:
    ReorderLayerDatasetInterleave8Block4()
    {
        add_config(TensorShape(16U, 9U), TensorShape(16U, 16U), WeightFormat::OHWI);
        add_config(TensorShape(16U, 16U), TensorShape(16U, 16U), WeightFormat::OHWI);
        add_config(TensorShape(16U, 511U), TensorShape(16U, 512U), WeightFormat::OHWI);
        add_config(TensorShape(248U, 301U), TensorShape(248U, 304U), WeightFormat::OHWI);
    }
};

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_DATASETS_REORDERLAYERDATASET_H
