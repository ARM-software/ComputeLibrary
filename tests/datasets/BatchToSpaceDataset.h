/*
 * Copyright (c) 2018-2019, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_BATCH_TO_SPACE_LAYER_DATASET
#define ARM_COMPUTE_TEST_BATCH_TO_SPACE_LAYER_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
class BatchToSpaceLayerDataset
{
public:
    using type = std::tuple<TensorShape, std::vector<int32_t>, CropInfo, TensorShape>;

    struct iterator
    {
        iterator(std::vector<TensorShape>::const_iterator          src_it,
                 std::vector<std::vector<int32_t>>::const_iterator block_shape_it,
                 std::vector<CropInfo>::const_iterator             crop_info_it,
                 std::vector<TensorShape>::const_iterator          dst_it)
            : _src_it{ std::move(src_it) },
              _block_shape_it{ std::move(block_shape_it) },
              _crop_info_it{ std::move(crop_info_it) },
              _dst_it{ std::move(dst_it) }
        {
        }

        std::string description() const
        {
            std::stringstream description;
            description << "In=" << *_src_it << ":";
            description << "BlockShape=" << *_block_shape_it << ":";
            description << "CropInfo=" << *_crop_info_it << ":";
            description << "Out=" << *_dst_it;
            return description.str();
        }

        BatchToSpaceLayerDataset::type operator*() const
        {
            return std::make_tuple(*_src_it, *_block_shape_it, *_crop_info_it, *_dst_it);
        }

        iterator &operator++()
        {
            ++_src_it;
            ++_block_shape_it;
            ++_crop_info_it;
            ++_dst_it;

            return *this;
        }

    private:
        std::vector<TensorShape>::const_iterator          _src_it;
        std::vector<std::vector<int32_t>>::const_iterator _block_shape_it;
        std::vector<CropInfo>::const_iterator             _crop_info_it;
        std::vector<TensorShape>::const_iterator          _dst_it;
    };

    iterator begin() const
    {
        return iterator(_src_shapes.begin(), _block_shapes.begin(), _crop_infos.begin(), _dst_shapes.begin());
    }

    int size() const
    {
        return std::min(std::min(std::min(_src_shapes.size(), _block_shapes.size()), _crop_infos.size()), _dst_shapes.size());
    }

    void add_config(const TensorShape &src, const std::vector<int32_t> &block_shape, const CropInfo &crop_info, const TensorShape &dst)
    {
        _src_shapes.emplace_back(std::move(src));
        _block_shapes.emplace_back(std::move(block_shape));
        _crop_infos.emplace_back(std::move(crop_info));
        _dst_shapes.emplace_back(std::move(dst));
    }

protected:
    BatchToSpaceLayerDataset()                            = default;
    BatchToSpaceLayerDataset(BatchToSpaceLayerDataset &&) = default;

private:
    std::vector<TensorShape>          _src_shapes{};
    std::vector<std::vector<int32_t>> _block_shapes{};
    std::vector<CropInfo>             _crop_infos{};
    std::vector<TensorShape>          _dst_shapes{};
};

/** Follow NCHW data layout across all datasets. I.e.
 * TensorShape(Width(X), Height(Y), Channel(Z), Batch(W))
 */

class SmallBatchToSpaceLayerDataset final : public BatchToSpaceLayerDataset
{
public:
    SmallBatchToSpaceLayerDataset()
    {
        // Block size = 1 (effectively no batch to space)
        add_config(TensorShape(1U, 1U, 1U, 4U), { 1U, 1U }, CropInfo(), TensorShape(1U, 1U, 1U, 4U));
        add_config(TensorShape(8U, 2U, 4U, 3U), { 1U, 1U }, CropInfo(), TensorShape(8U, 2U, 4U, 3U));
        // Same block size in both x and y
        add_config(TensorShape(3U, 2U, 1U, 4U), { 2U, 2U }, CropInfo(), TensorShape(6U, 4U, 1U, 1U));
        add_config(TensorShape(1U, 3U, 2U, 9U), { 3U, 3U }, CropInfo(), TensorShape(3U, 9U, 2U, 1U));
        // Different block size in x and y
        add_config(TensorShape(5U, 7U, 7U, 4U), { 2U, 1U }, CropInfo(), TensorShape(10U, 7U, 7U, 2U));
        add_config(TensorShape(3U, 3U, 1U, 8U), { 1U, 2U }, CropInfo(), TensorShape(3U, 6U, 1U, 4U));
        add_config(TensorShape(5U, 2U, 2U, 6U), { 3U, 2U }, CropInfo(), TensorShape(15U, 4U, 2U, 1U));
    }
};

/** Relative small shapes that are still large enough to leave room for testing cropping of the output shape
 */
class SmallBatchToSpaceLayerWithCroppingDataset final : public BatchToSpaceLayerDataset
{
public:
    SmallBatchToSpaceLayerWithCroppingDataset()
    {
        // Crop in both dims
        add_config(TensorShape(5U, 3U, 2U, 8U), { 2U, 2U }, CropInfo(1U, 1U, 2U, 1U), TensorShape(8U, 3U, 2U, 2U));
        // Left crop in x dim
        add_config(TensorShape(1U, 1U, 1U, 20U), { 4U, 5U }, CropInfo(2U, 1U, 0U, 2U), TensorShape(1U, 3U, 1U, 1U));
        // Left crop in y dim
        add_config(TensorShape(3U, 1U, 1U, 8U), { 2U, 4U }, CropInfo(0U, 0U, 2U, 1U), TensorShape(6U, 1U, 1U, 1U));
    }
};
class LargeBatchToSpaceLayerDataset final : public BatchToSpaceLayerDataset
{
public:
    LargeBatchToSpaceLayerDataset()
    {
        // Same block size in both x and y
        add_config(TensorShape(64U, 32U, 2U, 4U), { 2U, 2U }, CropInfo(), TensorShape(128U, 64U, 2U, 1U));
        add_config(TensorShape(128U, 16U, 2U, 18U), { 3U, 3U }, CropInfo(), TensorShape(384U, 48U, 2U, 2U));
        // Different block size in x and y
        add_config(TensorShape(16U, 8U, 2U, 8U), { 4U, 1U }, CropInfo(), TensorShape(64U, 8U, 2U, 2U));
        add_config(TensorShape(8U, 16U, 2U, 8U), { 2U, 4U }, CropInfo(), TensorShape(16U, 64U, 2U, 1U));
    }
};
} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BATCH_TO_SPACE_LAYER_DATASET */
