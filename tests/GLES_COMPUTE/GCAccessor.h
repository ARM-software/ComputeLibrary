/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_GCACCESSOR_H__
#define __ARM_COMPUTE_TEST_GCACCESSOR_H__

#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "tests/IAccessor.h"

namespace arm_compute
{
namespace test
{
/** Accessor implementation for @ref GCTensor objects. */
class GCAccessor : public IAccessor
{
public:
    /** Create an accessor for the given @p tensor.
     *
     * @param[in, out] tensor To be accessed tensor.
     *
     * @note The GLES memory is mapped by the constructor.
     *
     */
    GCAccessor(GCTensor &tensor);

    GCAccessor(const GCAccessor &) = delete;
    GCAccessor &operator=(const GCAccessor &) = delete;
    GCAccessor(GCAccessor &&)                 = default;
    GCAccessor &operator=(GCAccessor &&) = default;

    /** Destructor that unmaps the GLES memory. */
    ~GCAccessor();

    TensorShape      shape() const override;
    size_t           element_size() const override;
    size_t           size() const override;
    Format           format() const override;
    DataType         data_type() const override;
    int              num_channels() const override;
    int              num_elements() const override;
    PaddingSize      padding() const override;
    int              fixed_point_position() const override;
    QuantizationInfo quantization_info() const override;
    const void *operator()(const Coordinates &coord) const override;
    void *operator()(const Coordinates &coord) override;

private:
    GCTensor &_tensor;
};

inline GCAccessor::GCAccessor(GCTensor &tensor)
    : _tensor{ tensor }
{
    _tensor.map();
}

inline GCAccessor::~GCAccessor()
{
    _tensor.unmap();
}

inline TensorShape GCAccessor::shape() const
{
    return _tensor.info()->tensor_shape();
}

inline size_t GCAccessor::element_size() const
{
    return _tensor.info()->element_size();
}

inline size_t GCAccessor::size() const
{
    return _tensor.info()->total_size();
}

inline Format GCAccessor::format() const
{
    return _tensor.info()->format();
}

inline DataType GCAccessor::data_type() const
{
    return _tensor.info()->data_type();
}

inline int GCAccessor::num_channels() const
{
    return _tensor.info()->num_channels();
}

inline int GCAccessor::num_elements() const
{
    return _tensor.info()->tensor_shape().total_size();
}

inline PaddingSize GCAccessor::padding() const
{
    return _tensor.info()->padding();
}

inline int GCAccessor::fixed_point_position() const
{
    return _tensor.info()->fixed_point_position();
}

inline QuantizationInfo GCAccessor::quantization_info() const
{
    return _tensor.info()->quantization_info();
}

inline const void *GCAccessor::operator()(const Coordinates &coord) const
{
    return _tensor.ptr_to_element(coord);
}

inline void *GCAccessor::operator()(const Coordinates &coord)
{
    return _tensor.ptr_to_element(coord);
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_GCACCESSOR_H__ */
