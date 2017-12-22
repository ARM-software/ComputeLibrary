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
#ifndef __ARM_COMPUTE_TEST_CLACCESSOR_H__
#define __ARM_COMPUTE_TEST_CLACCESSOR_H__

#include "arm_compute/runtime/CL/CLTensor.h"
#include "tests/IAccessor.h"

namespace arm_compute
{
namespace test
{
/** Accessor implementation for @ref CLTensor objects. */
class CLAccessor : public IAccessor
{
public:
    /** Create an accessor for the given @p tensor.
     *
     * @param[in, out] tensor To be accessed tensor.
     *
     * @note The CL memory is mapped by the constructor.
     *
     */
    CLAccessor(CLTensor &tensor);

    CLAccessor(const CLAccessor &) = delete;
    CLAccessor &operator=(const CLAccessor &) = delete;
    CLAccessor(CLAccessor &&)                 = default;
    CLAccessor &operator=(CLAccessor &&) = default;

    /** Destructor that unmaps the CL memory. */
    ~CLAccessor();

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
    const void *data() const;
    void       *data();

private:
    CLTensor &_tensor;
};

inline CLAccessor::CLAccessor(CLTensor &tensor)
    : _tensor{ tensor }
{
    _tensor.map();
}

inline CLAccessor::~CLAccessor()
{
    _tensor.unmap();
}

inline TensorShape CLAccessor::shape() const
{
    return _tensor.info()->tensor_shape();
}

inline size_t CLAccessor::element_size() const
{
    return _tensor.info()->element_size();
}

inline size_t CLAccessor::size() const
{
    return _tensor.info()->total_size();
}

inline Format CLAccessor::format() const
{
    return _tensor.info()->format();
}

inline DataType CLAccessor::data_type() const
{
    return _tensor.info()->data_type();
}

inline int CLAccessor::num_channels() const
{
    return _tensor.info()->num_channels();
}

inline int CLAccessor::num_elements() const
{
    return _tensor.info()->tensor_shape().total_size();
}

inline PaddingSize CLAccessor::padding() const
{
    return _tensor.info()->padding();
}

inline int CLAccessor::fixed_point_position() const
{
    return _tensor.info()->fixed_point_position();
}

inline QuantizationInfo CLAccessor::quantization_info() const
{
    return _tensor.info()->quantization_info();
}

inline const void *CLAccessor::data() const
{
    return _tensor.buffer();
}

inline void *CLAccessor::data()
{
    return _tensor.buffer();
}

inline const void *CLAccessor::operator()(const Coordinates &coord) const
{
    return _tensor.ptr_to_element(coord);
}

inline void *CLAccessor::operator()(const Coordinates &coord)
{
    return _tensor.ptr_to_element(coord);
}
} // namespace test
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_CLACCESSOR_H__ */
