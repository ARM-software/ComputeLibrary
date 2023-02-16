/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_SUBTENSORINFO_H
#define ARM_COMPUTE_SUBTENSORINFO_H

#include "arm_compute/core/ITensorInfo.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorShape.h"

#include <cstddef>
#include <memory>

namespace arm_compute
{
/** Store the sub tensor's metadata */
class SubTensorInfo final : public ITensorInfo
{
public:
    /** Default constructor */
    SubTensorInfo();
    /** Default constructor
     *
     * @param[in] parent        Metadata of parent tensor.
     * @param[in] tensor_shape  Tensor shape. Shape must fit inside parent's shape.
     *                          X and Y dimensions must match the parent's ones.
     * @param[in] coords        Coordinates of starting element inside parent tensor.
     * @param[in] extend_parent (Optional) Extend parent with subtensor shape if subtensor indexes out of bounds
     */
    SubTensorInfo(ITensorInfo *parent, TensorShape tensor_shape, Coordinates coords, bool extend_parent = false);
    /** Default destructor */
    ~SubTensorInfo() = default;
    /** Allow instances of this class to be copy constructed */
    SubTensorInfo(const SubTensorInfo &) = default;
    /** Allow instances of this class to be copied */
    SubTensorInfo &operator=(const SubTensorInfo &) = default;
    /** Allow instances of this class to be move constructed */
    SubTensorInfo(SubTensorInfo &&) = default;
    /** Allow instances of this class to be moved */
    SubTensorInfo &operator=(SubTensorInfo &&) = default;
    /** Returns the coordinates of the sub-tensor inside the parent tensor
     *
     * @return Sub-tensor coordinates
     */
    Coordinates coords() const
    {
        return _coords;
    }

    // Inherited methods overridden:
    std::unique_ptr<ITensorInfo> clone() const override;
    ITensorInfo &set_data_type(DataType data_type) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_data_type(data_type);
        return *this;
    };
    ITensorInfo &set_data_layout(const DataLayout &data_layout) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_data_layout(data_layout);
        return *this;
    };
    ITensorInfo &set_num_channels(int num_channels) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_num_channels(num_channels);
        return *this;
    };
    ITensorInfo &set_format(Format format) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_format(format);
        return *this;
    };
    ITensorInfo &set_tensor_shape(const TensorShape &shape) override;
    ITensorInfo &set_tensor_dims_state(const TensorDimsState &state) override;
    ITensorInfo &set_quantization_info(const QuantizationInfo &quantization_info) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_quantization_info(quantization_info);
        return *this;
    }
    ITensorInfo &reset_padding() override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->reset_padding();
        return *this;
    }
    bool auto_padding() override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->auto_padding();
    };

    ITensorInfo &set_lock_paddings(bool flag) override;

    bool lock_paddings() const override;

    bool extend_padding(const PaddingSize &padding) override;

    size_t dimension(size_t index) const override
    {
        return _tensor_shape[index];
    }
    size_t dimension(DataLayoutDimension dimension) const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return get_data_layout_dimension_index(_parent->data_layout(), dimension);
    }
    const Strides &strides_in_bytes() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->strides_in_bytes();
    }
    size_t offset_first_element_in_bytes() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->offset_element_in_bytes(_coords);
    }
    int32_t offset_element_in_bytes(const Coordinates &pos) const override;
    size_t element_size() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->element_size();
    }
    size_t num_dimensions() const override
    {
        return _tensor_shape.num_dimensions();
    }
    size_t num_channels() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->num_channels();
    }
    const TensorShape &tensor_shape() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _tensor_shape;
    }
    const TensorDimsState &tensor_dims_state() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _dims_state;
    }
    DataType data_type() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->data_type();
    }
    Format format() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->format();
    }
    size_t total_size() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->total_size();
    }
    PaddingSize padding() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->padding();
    }
    bool has_padding() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->has_padding();
    }
    bool is_resizable() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->is_resizable();
    }
    bool is_dynamic() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->is_dynamic();
    }
    bool are_values_constant() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->are_values_constant();
    }
    ITensorInfo &set_is_resizable(bool is_resizable) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_is_resizable(is_resizable);
        return *this;
    }
    ITensorInfo &set_are_values_constant(bool are_values_constant) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_are_values_constant(are_values_constant);
        return *this;
    }
    ValidRegion valid_region() const override
    {
        return _valid_region;
    }
    void set_valid_region(const ValidRegion &valid_region) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        // Check if subtensor is valid if parent is configured
        if(_parent->tensor_shape().total_size() != 0)
        {
            ARM_COMPUTE_ERROR_ON_INVALID_SUBTENSOR_VALID_REGION(_parent->valid_region(), valid_region);
        }
        _valid_region = valid_region;
    }
    QuantizationInfo quantization_info() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->quantization_info();
    }
    DataLayout data_layout() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->data_layout();
    }
    ITensorInfo::Id id() const override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        return _parent->id();
    }
    ITensorInfo &set_id(ITensorInfo::Id id) override
    {
        ARM_COMPUTE_ERROR_ON(_parent == nullptr);
        _parent->set_id(id);
        return *this;
    }

private:
    ITensorInfo    *_parent;
    TensorShape     _tensor_shape;
    TensorDimsState _dims_state;
    Coordinates     _coords;
    ValidRegion     _valid_region;
    bool            _extend_parent;
    bool            _lock_paddings;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_SUBTENSORINFO_H */
