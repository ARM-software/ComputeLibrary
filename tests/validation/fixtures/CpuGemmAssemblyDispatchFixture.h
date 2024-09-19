/*
 * Copyright (c) 2017-2024 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMASSEMBLYDISPATCHFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMASSEMBLYDISPATCHFIXTURE_H

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/GEMM.h"
#include "arm_compute/core/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuGemmAssemblyDispatchGenericValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape_a,
               TensorShape shape_b,
               TensorShape shape_c,
               TensorShape output_shape,
               float       alpha,
               float       beta,
               DataType    data_type,
               bool        accumulate)
    {
        if(std::is_same<TensorType, Tensor>::value &&  // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }
        ARM_COMPUTE_UNUSED(alpha);
        ARM_COMPUTE_UNUSED(beta);
        _target    = compute_target(shape_a, shape_b, shape_c, output_shape, data_type, accumulate);
        _reference = compute_reference(shape_a, shape_b, output_shape, data_type, accumulate);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float lo = -1.f, float hi = 1.f)
    {
        switch (tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{float(lo), float(hi)};
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(lo, hi);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &shape_a,
                              const TensorShape &shape_b,
                              const TensorShape &shape_c,
                              const TensorShape &output_shape,
                              DataType           data_type,
                              bool               accumulate)
    {
        ARM_COMPUTE_UNUSED(shape_c);
        // Create tensors
        TensorType  a   = create_tensor<TensorType>(shape_a, data_type, 1);
        TensorType  b   = create_tensor<TensorType>(shape_b, data_type, 1);
        TensorType *c   = nullptr;
        TensorType  dst = create_tensor<TensorType>(output_shape, data_type, 1);

        // Create and configure function
        FunctionType gemm;

        add_padding_x({&a, &b, &dst});

        GEMMInfo gemm_info;
        gemm_info.set_accumulate(accumulate);

        ARM_COMPUTE_ASSERT(gemm.validate(a.info(), b.info(), nullptr, dst.info(), gemm_info));

        // The GEMMinfo includes the values of the depth in case of reinterpreted 3d output.
        // If the output shape has the same number of dimensions of the input the method called is a 2D matrix multiplication (depth_output_reinterpreted_as_3D = 0),
        // in the other case we have to use the reinterpreted version of GEMM (depth_output_reinterpreted_as_3D = depth of the 3D output).
        gemm.configure(a.info(), b.info(), nullptr, dst.info(), gemm_info);

        ARM_COMPUTE_ASSERT(gemm.is_configured());

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(a), 0, -1.f, 1.f);
        fill(AccessorType(b), 1, -1.f, 1.f);
        if (accumulate)
        {
            fill(AccessorType(dst), 6);
        };

        ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &a},
                             {arm_compute::TensorType::ACL_SRC_1, &b},
                             {arm_compute::TensorType::ACL_SRC_2, c},
                             {arm_compute::TensorType::ACL_DST_0, &dst}};

        // Prepare memory
        ITensorPack prep_pack{{arm_compute::TensorType::ACL_SRC_1, &b}, {arm_compute::TensorType::ACL_SRC_2, c}};

        experimental::MemoryRequirements aux_mem_req = gemm.workspace();
        MemoryGroup                      memory_group{};

        WorkspaceData<Tensor> workspace = manage_workspace<Tensor>(aux_mem_req, memory_group, run_pack, prep_pack);

        gemm.prepare(prep_pack);
        MemoryGroupResourceScope scope_mg(memory_group);

        auto has_reshape = std::find_if(aux_mem_req.begin(), aux_mem_req.end(),
                                        [](const arm_compute::experimental::MemoryInfo &m) -> bool {
                                            return m.lifetime == arm_compute::experimental::MemoryLifetime::Persistent;
                                        });

        if (has_reshape != std::end(aux_mem_req))
        {
            b.mark_as_unused();
        }
        else
        {
            run_pack.add_const_tensor(ACL_SRC_1, &b);
        }

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(aux_mem_req, workspace);
        // End of preparing

        // Compute GEMM function
        gemm.run(run_pack);

        a.allocator()->free();
        b.allocator()->free();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape_a,
                                      const TensorShape &shape_b,
                                      const TensorShape &output_shape,
                                      DataType           data_type,
                                      bool               accumulate)
    {
        // Create reference
        SimpleTensor<T> a{shape_a, data_type, 1};
        SimpleTensor<T> b{shape_b, data_type, 1};
        SimpleTensor<T> c{output_shape, data_type, 1};
        SimpleTensor<T> dst{output_shape, data_type, 1};

        // Fill reference
        fill(a, 0, -1.f, 1.f);
        fill(b, 1, -1.f, 1.f);
        fill(c, 2);

        // Do in place summation
        if (accumulate)
        {
            fill(dst, 6);
        }

        // Setting beta to 0 will effectively disable C for the
        // computation of the reference: A * B + 0 * C
        // Use transposed tensors if boolean enabled else use original tensors
        if (accumulate)
        {
            reference::gemm_accumulate<T>(a, b, c, 1.0f, 0.f, dst);
            return dst;
        }
        else
        {
            return reference::gemm<T>(a, b, c, 1.f, 0.f);
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool accumulate>
class CpuGemmAssemblyDispatchValidationFixture
    : protected CpuGemmAssemblyDispatchGenericValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape_a,
               TensorShape shape_b,
               TensorShape shape_c,
               TensorShape output_shape,
               float       alpha,
               float       beta,
               DataType    data_type)
    {
        CpuGemmAssemblyDispatchGenericValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type, accumulate);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMASSEMBLYDISPATCHFIXTURE_H
