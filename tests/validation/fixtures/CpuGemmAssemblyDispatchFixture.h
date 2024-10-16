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

#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/functions/NEReorderLayer.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"

#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/GEMM.h"

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
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         shape_c,
               TensorShape         output_shape,
               float               alpha,
               float               beta,
               DataType            data_type,
               bool                accumulate,
               bool                pretranspose_b,
               ActivationLayerInfo act_info)
    {
        if(std::is_same<TensorType, Tensor>::value &&  // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }
        ARM_COMPUTE_UNUSED(alpha);
        ARM_COMPUTE_UNUSED(beta);
        _target =
            compute_target(shape_a, shape_b, shape_c, output_shape, data_type, accumulate, pretranspose_b, act_info);
        _reference = compute_reference(shape_a, shape_b, output_shape, data_type, accumulate, act_info);
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

    TensorType compute_target(const TensorShape  &shape_a,
                              const TensorShape  &shape_b,
                              const TensorShape  &shape_c,
                              const TensorShape  &output_shape,
                              DataType            data_type,
                              bool                accumulate,
                              bool                pretranspose_b,
                              ActivationLayerInfo act_info)
    {
        ARM_COMPUTE_UNUSED(shape_c);
        // Create tensors
        TensorType  a            = create_tensor<TensorType>(shape_a, data_type, 1);
        TensorType  b            = create_tensor<TensorType>(shape_b, data_type, 1);
        TensorType  b_transposed = create_tensor<TensorType>({shape_b[1], shape_b[0]}, data_type, 1);
        TensorType *c            = nullptr;
        TensorType  dst          = create_tensor<TensorType>(output_shape, data_type, 1);

        // Create and configure function
        FunctionType gemm;
        NETranspose  transpose;

        add_padding_x({&a, &b, &b_transposed, &dst});

        GEMMInfo gemm_info;
        gemm_info.set_accumulate(accumulate);
        gemm_info.set_pretranspose_B(pretranspose_b);
        gemm_info.set_activation_info(act_info);

        TensorType &b_to_use = pretranspose_b ? b_transposed : b;

        ARM_COMPUTE_ASSERT(gemm.validate(a.info(), b_to_use.info(), nullptr, dst.info(), gemm_info));

        gemm.configure(a.info(), b_to_use.info(), nullptr, dst.info(), gemm_info);

        ARM_COMPUTE_ASSERT(gemm.is_configured());

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b_transposed.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        b_transposed.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b_transposed.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(a), 0, -1.f, 1.f);
        fill(AccessorType(b), 1, -1.f, 1.f);
        if (accumulate)
        {
            fill(AccessorType(dst), 6);
        };

        if (pretranspose_b)
        {
            transpose.configure(&b, &b_transposed);
            transpose.run();
        }

        ITensorPack run_pack{{arm_compute::TensorType::ACL_SRC_0, &a},
                             {arm_compute::TensorType::ACL_SRC_1, &b_to_use},
                             {arm_compute::TensorType::ACL_SRC_2, c},
                             {arm_compute::TensorType::ACL_DST_0, &dst}};

        // Prepare memory
        ITensorPack prep_pack{{arm_compute::TensorType::ACL_SRC_1, &b_to_use}, {arm_compute::TensorType::ACL_SRC_2, c}};

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
            run_pack.add_const_tensor(ACL_SRC_1, &b_to_use);
        }

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(aux_mem_req, workspace);
        // End of preparing

        // Compute GEMM function
        gemm.run(run_pack);

        a.allocator()->free();
        b.allocator()->free();
        b_transposed.allocator()->free();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape  &shape_a,
                                      const TensorShape  &shape_b,
                                      const TensorShape  &output_shape,
                                      DataType            data_type,
                                      bool                accumulate,
                                      ActivationLayerInfo act_info)
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

        if (accumulate)
        {
            reference::gemm_accumulate<T>(a, b, c, 1.0f, 0.f, dst);
        }
        else
        {
            dst = reference::gemm<T>(a, b, c, 1.f, 0.f);
        }

        if (act_info.enabled())
        {
            return reference::activation_layer<T>(dst, act_info);
        }
        return dst;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuGemmAssemblyDispatchValidationFixture
    : protected CpuGemmAssemblyDispatchGenericValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         shape_c,
               TensorShape         output_shape,
               float               alpha,
               float               beta,
               DataType            data_type,
               bool                accumulate,
               bool                pretranspose_b,
               ActivationLayerInfo act_info)
    {
        CpuGemmAssemblyDispatchGenericValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape_a, shape_b, shape_c, output_shape, alpha, beta, data_type, accumulate, pretranspose_b, act_info);
    }
};

#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuGemmAssemblyDispatchFixedFormatFixture
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
        ARM_COMPUTE_UNUSED(alpha);
        ARM_COMPUTE_UNUSED(beta);
        this->_target = compute_target(shape_a, shape_b, shape_c, output_shape, data_type);
        this->_reference =
            this->compute_reference(shape_a, shape_b, output_shape, data_type, false, ActivationLayerInfo());
    }

protected:
    inline TensorInfo prepare_weights(const TensorInfo tensor_info, const arm_compute::WeightFormat weight_format)
    {
        const DataLayout  data_layout  = tensor_info.data_layout();
        const DataType    data_type    = tensor_info.data_type();
        const TensorShape tensor_shape = tensor_info.tensor_shape();
        const int N = tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES)]; // N=O
        const int H = tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT)];
        const int W = tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH)];
        const int C = tensor_shape[get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL)]; // C=I

        const int interleave_by = arm_compute::interleave_by(weight_format);
        const int block_by      = arm_compute::block_by(weight_format);
        const int Ip            = arm_gemm::roundup<unsigned int>(C, block_by);      // C'=I'
        const int Op            = arm_gemm::roundup<unsigned int>(N, interleave_by); // O'=N'

        arm_compute::Strides strides_in_bytes = tensor_info.strides_in_bytes();
        strides_in_bytes.set(1, Ip * interleave_by * W * tensor_info.element_size());
        strides_in_bytes.set(2, Op * interleave_by * W * tensor_info.element_size());

        const size_t offset_first_element_in_bytes = tensor_info.offset_first_element_in_bytes();

        // Total size needs to include padded dimensions
        const size_t total_size_in_bytes = Op * H * W * Ip * tensor_info.element_size();

        const TensorShape TS({tensor_shape[0], arm_compute::ceil_to_multiple<int32_t, int32_t>(tensor_shape[1], 4)});

        TensorInfo new_tensor_info = tensor_info;
        new_tensor_info.set_data_layout(DataLayout::UNKNOWN);
        new_tensor_info.init(TS, tensor_info.num_channels(), data_type, strides_in_bytes, offset_first_element_in_bytes,
                             total_size_in_bytes);
        return new_tensor_info;
    }

    TensorType compute_target(
        TensorShape shape_a, TensorShape shape_b, TensorShape shape_c, TensorShape output_shape, DataType data_type)
    {
        ARM_COMPUTE_UNUSED(shape_c);
        permute(shape_b, PermutationVector(1U, 0U));
        // Create tensors
        TensorType a   = create_tensor<TensorType>(shape_a, data_type, 1, QuantizationInfo(), DataLayout::NCHW);
        TensorType b   = create_tensor<TensorType>(shape_b, data_type, 1, QuantizationInfo(), DataLayout::NCHW);
        TensorType c   = nullptr;
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo(), DataLayout::NCHW);

        // Create and configure function
        FunctionType              gemm;
        NEReorderLayer            reorder;
        arm_compute::WeightFormat computed_weight_format{arm_compute::WeightFormat::ANY};
        GEMMInfo                  gemm_info;

        gemm_info.set_fixed_format(true);
        gemm_info.set_accumulate(false);
        gemm_info.set_weight_format(computed_weight_format);

        const bool kernel_found = bool(
            FunctionType::has_opt_impl(computed_weight_format, a.info(), b.info(), nullptr, dst.info(), gemm_info));

        ARM_COMPUTE_ASSERT(kernel_found);
        gemm_info.set_weight_format(computed_weight_format);
        gemm_info.set_fast_math(is_fixed_format_fast_math(computed_weight_format));
        TensorType b_transformed = create_tensor<TensorType>(prepare_weights(*b.info(), computed_weight_format));

        a.info()->set_are_values_constant(false);
        b_transformed.info()->set_are_values_constant(false);

        ARM_COMPUTE_ASSERT(a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(b_transformed.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        b_transformed.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!a.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!b_transformed.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        this->fill(AccessorType(a), 0, -1.f, 1.f);
        this->fill(AccessorType(b), 1, -1.f, 1.f);

        // Reorder weight to the expected format
        reorder.configure(&b, &b_transformed, WeightFormat::OHWI, computed_weight_format);
        reorder.run();

        ARM_COMPUTE_ASSERT(gemm.validate(a.info(), b_transformed.info(), nullptr, dst.info(), gemm_info));
        gemm.configure(a.info(), b_transformed.info(), nullptr, dst.info(), gemm_info);
        ARM_COMPUTE_ASSERT(gemm.is_configured());

        ITensorPack run_pack;
        run_pack.add_const_tensor(arm_compute::TensorType::ACL_SRC_0, &a);
        run_pack.add_const_tensor(arm_compute::TensorType::ACL_SRC_1, &b_transformed);
        run_pack.add_tensor(arm_compute::TensorType::ACL_SRC_2, &c);
        run_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst);

        // Prepare memory
        ITensorPack prep_pack{{arm_compute::TensorType::ACL_SRC_1, &b_transformed},
                              {arm_compute::TensorType::ACL_SRC_2, &c}};

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
            b_transformed.mark_as_unused();
        }
        else
        {
            run_pack.add_const_tensor(ACL_SRC_1, &b_transformed);
        }

        // Release temporary tensors that are only used in prepare stage
        release_temporaries<Tensor>(aux_mem_req, workspace);
        // End of preparing

        gemm.run(run_pack);

        a.allocator()->free();
        b.allocator()->free();
        b_transformed.allocator()->free();

        return dst;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    bool            _kernel_found{false};
};

#endif //ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMASSEMBLYDISPATCHFIXTURE_H
