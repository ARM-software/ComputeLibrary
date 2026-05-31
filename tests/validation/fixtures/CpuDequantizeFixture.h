/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUDEQUANTIZEFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUDEQUANTIZEFIXTURE_H

#include "tests/validation/fixtures/DequantizationLayerFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuDequantizationValidationFixture
    : public DequantizationValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, DataType src_data_type, DataType dst_datatype, DataLayout data_layout)
    {
        if (!cpu_supports_dtypes({src_data_type, dst_datatype}))
        {
            return;
        }

        this->_quantization_info = this->generate_quantization_info(src_data_type, shape.z());
        this->_target            = this->compute_target(shape, src_data_type, dst_datatype, data_layout);
        this->_reference         = this->compute_reference(shape, src_data_type);
    }

protected:
    TensorType compute_target(TensorShape shape, DataType src_data_type, DataType dst_datatype, DataLayout data_layout)
    {
        if (data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, src_data_type, 1, this->_quantization_info, data_layout);
        TensorType dst = create_tensor<TensorType>(shape, dst_datatype, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType dequantization_layer;
        dequantization_layer.configure(src.info(), dst.info());

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        this->fill(AccessorType(src));

        // Prepare tensor pack
        ITensorPack run_pack = {{arm_compute::TensorType::ACL_SRC, &src}, {arm_compute::TensorType::ACL_DST, &dst}};

        // Compute function
        dequantization_layer.run(run_pack);

        return dst;
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUDEQUANTIZEFIXTURE_H
