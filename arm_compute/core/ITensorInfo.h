/*
 * Copyright (c) 2016-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_ITENSORINFO_H
#define ARM_COMPUTE_ITENSORINFO_H

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Strides.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "support/ICloneable.h"

#include <cstddef>

namespace arm_compute
{
// Note: Any changes to the fields of the class below that have setters should be mirrored
// (if possible) in the auto_init_if_empty function in AutoConfiguration.h

/** Store the tensor's metadata */
class ITensorInfo : public misc::ICloneable<ITensorInfo>
{
public:
    using TensorDimsState = std::vector<int>;
    /** An id that uniquely identifies an ITensorInfo within some domain (e.g. a workload)
     */
    using Id = int32_t;
    /** An invalid tensor id within a domain */
    static constexpr Id invalid_tensor_id = 0;
    /** Get the value representing dynamic dimension state
     *
     * @return Value representing dynamic dimension state
     *
     */
    static constexpr int32_t get_dynamic_state_value()
    {
        return _dynamic_dimension;
    }
    /** Get the value representing static dimension state
     *
     * @return Value representing static dimension state
     *
     */
    static constexpr int32_t get_static_state_value()
    {
        return _static_dimension;
    }
    /** Default virtual destructor */
    virtual ~ITensorInfo() = default;
    /** Set the data type to the specified value.
     *
     * @warning This resets the format to UNKNOWN.
     *
     * @param[in] data_type The new data type.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_data_type(DataType data_type) = 0;
    /** Set the number of channels to the specified value.
     *
     * @warning This resets the format to UNKNOWN.
     *
     * @param[in] num_channels New number of channels.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_num_channels(int num_channels) = 0;
    /** Set the format of an already initialized tensor.
     *
     * @note If the data type has already been configured (i.e. not UNKNOWN) it
     * must match the new format. If data type hasn't been configured it will
     * be based on the format.
     *
     * @param[in] format Single-plane format of the tensor.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_format(Format format) = 0;
    /** Set the shape of an already initialized tensor.
     *
     * @warning Changing the shape requires to recompute the strides and is
     * therefore only possible if the tensor hasn't been allocated yet.
     *
     * @param[in] shape New tensor shape.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_tensor_shape(const TensorShape &shape) = 0;
    /** Set the state for each dimension of the tensor
     *
     * This sets the state of each dimension of the shape in terms of dynamic behavior using -1 where appropriate.
     * The index in the state is a 1 to 1 mapping with the shape dimension index.
     * For example if you want to express [?, 3, 3] as a dynamic input then [-1, 3, 3] has to be set as a state
     *
     * @param[in] state Tensor dimensions state
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_tensor_dims_state(const TensorDimsState &state) = 0;
    /** Set the quantization settings (scale and offset) of the tensor.
     *
     * @param[in] quantization_info QuantizationInfo containing the scale and offset
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_quantization_info(const QuantizationInfo &quantization_info) = 0;
    /** Set the data layout of the tensor.
     *
     * @param[in] data_layout DataLayout containing the layout data information.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_data_layout(const DataLayout &data_layout) = 0;
    /** Resets the padding settings of the tensor.
    *
    * @return Reference to this ITensorInfo object
    */
    virtual ITensorInfo &reset_padding() = 0;
    /** Update the offset to the first element and the strides to automatically computed values.
     *
     * @note The padding used by this method is really conservative so that the tensor can be used for most functions.
     *
     * @return True if the strides or the offset to the first element have changed.
     */
    virtual bool auto_padding() = 0;
    /** Set the lock paddings flag of the tensor.
     * It should be set to True, when the tensor could be mapped to camera or frame buffer.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_lock_paddings(bool flag) = 0;
    /** Get the lock paddings flag value
     *
     * @return lock paddings flag value
     */
    virtual bool lock_paddings() const = 0;
    /** Update the offset to the first element, the strides and the total size.
     *
     * @note This function can only increase the offset, strides and total size.
     *
     * @param[in] padding Padding around the XY plane in number of elements.
     *
     * @return True if the strides, offset and total size have changed.
     */
    virtual bool extend_padding(const PaddingSize &padding) = 0;
    /** Return the size of the requested dimension
     *
     * @param[in] index Index of the dimension
     *
     * @return Dimension of the requested dimension
     */
    virtual size_t dimension(size_t index) const = 0;
    /** Return the size of the requested data layout dimension
     *
     * @param[in] dimension DataLayoutDimension of the dimension
     *
     * @return Dimension of the requested dimension
     */
    virtual size_t dimension(DataLayoutDimension dimension) const = 0;
    /** The strides in bytes for accessing each dimension of the tensor
     *
     * @return Strides in bytes for each tensor dimension
     */
    virtual const Strides &strides_in_bytes() const = 0;
    /** The offset from the beginning of the memory allocation to the first element of the tensor.
     *  This can be used to access efficiently elements in a 2D tensor
     *
     * @return The offset in bytes to access the first element of the tensor.
     */
    virtual size_t offset_first_element_in_bytes() const = 0;
    /** The offset in bytes from the beginning of the memory allocation to access the element at position (x, y, z ...)
     *
     * @param[in] pos Vector with the coordinates of the element to access.
     *                The size of this vector must be equal to the number of dimensions of the tensor
     *
     * @return Offset in bytes from the beginning of the memory allocation to access the element (x, y, z, ...)
     */
    virtual int32_t offset_element_in_bytes(const Coordinates &pos) const = 0;

    /** Element size in bytes calculated as data_size() * num_channels()
     *
     * @return The size of one element in bytes
     */
    virtual size_t element_size() const = 0;
    /** The number of dimensions of the tensor (rank)
     *
     * @return The number of dimensions of the tensor (rank)
     */
    virtual size_t num_dimensions() const = 0;
    /** The number of channels for each tensor element
     *
     * @return The number of channels for each tensor element
     */
    virtual size_t num_channels() const = 0;
    /** Size for each dimension of the tensor
     *
     * @return A vector with the size for each dimension of the tensor
     */
    virtual const TensorShape &tensor_shape() const = 0;
    /** State of each dimension of the tensor shape
     *
     * @return A vector with the state for each dimension of the tensor, where -1 specifies dynamic dimension
     */
    virtual const TensorDimsState &tensor_dims_state() const = 0;
    /** Data type used for each element of the tensor
     *
     * @return Tensor data type
     */
    virtual DataType data_type() const = 0;
    /** Colour format of the image
     *
     * @return Colour format of the image
     */
    virtual Format format() const = 0;
    /** Returns the total size of the tensor in bytes.
     *
     * @return Total size of the tensor in bytes.
     */
    virtual size_t total_size() const = 0;
    /** Padding of tensor.
     *
     * @return Padding.
     */
    virtual PaddingSize padding() const = 0;
    /** Checks if the tensor has been allocated with padding or not.
     *
     * @return True if padding is allocated in the tensor, otherwise false.
     */
    virtual bool has_padding() const = 0;
    /** Flag indicating whether the size of the tensor can be changed.
     *
     * @return True if the tensor size can be changed.
     */
    virtual bool is_resizable() const = 0;
    /** Flag indicating whether the shape of the tensor is dynamic, meaning that it can change on kernel/function execution.
     *
     * @return True if its dynamic else false
     */
    virtual bool is_dynamic() const = 0;
    /** Flag indicating whether the values of the tensor are constant, meaning that they can change on kernel/function execution.
     *
     * @return True if values are constant else false
     */
    virtual bool are_values_constant() const = 0;
    /** Set the flag whether the tensor size can be changed.
     *
     * @param[in] is_resizable Flag that marks the tensor if it can be changed or not.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_is_resizable(bool is_resizable) = 0;
    /** Set the flag whether the tensor values can change during kernel/function execution.
     *
     * @param[in] are_values_constant Flag that marks the tensor values if they can be changed or not.
     *
     * @return Reference to this ITensorInfo object
     */
    virtual ITensorInfo &set_are_values_constant(bool are_values_constant) = 0;
    /** Valid region of the tensor. All elements in the valid region have defined values, i.e. are not undefined.
     *
     * @return The valid region.
     */
    virtual ValidRegion valid_region() const = 0;
    /** Set the valid region of the tensor.
     *
     * @param[in] valid_region Valid region to set.
     */
    virtual void set_valid_region(const ValidRegion &valid_region) = 0;

    /** Get the quantization settings (scale and offset) of the tensor.
    *
    * @return A QuantizationInfo containing the scale and offset.
    */
    virtual QuantizationInfo quantization_info() const = 0;
    /** Get the data layout of the tensor.
    *
    * @return A DataLayout containing the layout data information.
    */
    virtual DataLayout data_layout() const = 0;
    /** Get the workload tensor id of the tensor.
    *
    * @return Workload tensor id of the tensor
    */
    virtual Id id() const = 0;
    /** Set the tensor id
    */
    virtual ITensorInfo &set_id(ITensorInfo::Id id) = 0;
    /** Check if the tensor id is valid
     */
    bool has_valid_id() const
    {
        return id() != invalid_tensor_id;
    }
    /** If infos are broadcast compatible tensor info's, return the broadcasted shape and the intersection of
     * the broadcasted valid regions of the tensors.
     *
     * Two tensor info's are broadcast compatible if their shapes are broadcast compatible.
     *
     * Two tensor shapes are broadcast compatible if for each dimension, they're equal or one of them is 1.
     *
     * If two shapes are compatible, each dimension in the broadcasted shape is the max of the original dimensions.
     *
     * @param[in] infos Tensor info's.
     *
     * @return The broadcasted shape and valid region, or an empty shape and valid region if the info's are
     * not broadcast compatible.
     */
    template <typename... Infos>
    static std::pair<TensorShape, ValidRegion> broadcast_shape_and_valid_region(const Infos &... infos)
    {
        TensorShape bc_shape = TensorShape::broadcast_shape(infos.tensor_shape()...);
        ValidRegion bc_valid_region{ Coordinates(), bc_shape };

        auto broadcast_valid_region = [&bc_valid_region](const ITensorInfo & info)
        {
            if(info.num_dimensions() != 0)
            {
                for(size_t d = 0; d < bc_valid_region.shape.num_dimensions(); ++d)
                {
                    const bool is_broadcast = (info.tensor_shape()[d] == 1);

                    const int    anchor_max = std::max(bc_valid_region.anchor[d], info.valid_region().anchor[d]);
                    const size_t valid_min  = std::min(bc_valid_region.shape[d], info.valid_region().shape[d]);

                    if(!is_broadcast || (valid_min == 0))
                    {
                        bc_valid_region.anchor.set(d, anchor_max);
                        bc_valid_region.shape.set(d, valid_min);
                    }
                }
            }
        };

        utility::for_each(broadcast_valid_region, infos...);

        return std::pair<TensorShape, ValidRegion>(bc_shape, bc_valid_region);
    }

private:
    static constexpr int32_t _dynamic_dimension = -1;
    static constexpr int32_t _static_dimension  = 0;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_TENSORINFO_H */
