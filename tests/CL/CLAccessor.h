/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_CLACCESSOR_H
#define ARM_COMPUTE_TEST_CLACCESSOR_H

#include "arm_compute/runtime/CL/CLTensor.h"
#include "tests/IAccessor.h"
#include "tests/framework/Framework.h"

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

    /** Prevent instances of this class from being copy constructed */
    CLAccessor(const CLAccessor &) = delete;
    /** Prevent instances of this class from being copied */
    CLAccessor &operator=(const CLAccessor &) = delete;
    /** Allow instances of this class to be move constructed */
    CLAccessor(CLAccessor &&) = default;

    /** Destructor that unmaps the CL memory. */
    ~CLAccessor();

    /** Get the tensor data.
     *
     * @return a constant pointer to the tensor data.
     */
    const void *data() const;
    /** Get the tensor data.
     *
     * @return a pointer to the tensor data.
     */
    void *data();

    // Inherited method overrides
    TensorShape      shape() const override;
    size_t           element_size() const override;
    size_t           size() const override;
    Format           format() const override;
    DataLayout       data_layout() const override;
    DataType         data_type() const override;
    int              num_channels() const override;
    int              num_elements() const override;
    PaddingSize      padding() const override;
    QuantizationInfo quantization_info() const override;
    const void *operator()(const Coordinates &coord) const override;
    void *operator()(const Coordinates &coord) override;

private:
    CLTensor &_tensor;
};

inline CLAccessor::CLAccessor(CLTensor &tensor)
    : _tensor{ tensor }
{
    if(!framework::Framework::get().configure_only() || !framework::Framework::get().new_fixture_call())
    {
        _tensor.map();
    }
}

inline CLAccessor::~CLAccessor()
{
    if(!framework::Framework::get().configure_only() || !framework::Framework::get().new_fixture_call())
    {
        _tensor.unmap();
    }
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

inline DataLayout CLAccessor::data_layout() const
{
    return _tensor.info()->data_layout();
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
#endif /* ARM_COMPUTE_TEST_CLACCESSOR_H */
