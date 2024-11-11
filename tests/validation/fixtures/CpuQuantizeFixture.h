/*
 * Copyright (c) 2017-2021, 2023-2024 Arm Limited.
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

#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUQUANTIZEFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUQUANTIZEFIXTURE_H

#include "tests/validation/fixtures/QuantizationLayerFixture.h"
#include "tests/validation/Helpers.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "tests/validation/Helpers.h"
namespace arm_compute
{
namespace test
{
namespace validation
{

template <typename TensorType, typename AccessorType, typename FunctionType, typename Tin, typename Tout>
class CpuQuantizationValidationFixture : public QuantizationValidationFixture<TensorType, AccessorType, FunctionType, Tin, Tout>
{
public:
void setup(TensorShape shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo)
{
    QuantizationInfo qinfo_in;
    if(std::is_same<TensorType, Tensor>::value &&  // Cpu
        (data_type_in == DataType::F16 || data_type_out == DataType::F16) && !CPUInfo::get().has_fp16())
    {
        return;
    }

    if(!cpu_supports_dtypes({data_type_in, data_type_out})) {
        return;
    }

    this->_target    = compute_target(shape, data_type_in, data_type_out, qinfo, qinfo_in);
    this->_reference = this->compute_reference(shape, data_type_in, data_type_out, qinfo, qinfo_in);
}

protected:
    TensorType compute_target(const TensorShape &shape, DataType data_type_in, DataType data_type_out, QuantizationInfo qinfo, QuantizationInfo qinfo_in)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type_in, 1, qinfo_in);
        TensorType dst = create_tensor<TensorType>(shape, data_type_out, 1, qinfo);

        // Create and configure function
        FunctionType quantization_layer;
        quantization_layer.configure(src.info(), dst.info());

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
        ITensorPack run_pack = { { arm_compute::TensorType::ACL_SRC, &src },
                                { arm_compute::TensorType::ACL_DST, &dst } };
        auto mg = MemoryGroup{};
        auto ws = arm_compute::manage_workspace<TensorType>(quantization_layer.workspace(), mg, run_pack);
        allocate_tensors(quantization_layer.workspace(), ws);

        // Compute function
        quantization_layer.run(run_pack);

        return dst;
    }
};


} // namespace validation
} // namespace test
} // namespace arm_compute


#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUQUANTIZEFIXTURE_H
