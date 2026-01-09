/*
 * Copyright (c) 2017-2026 Arm Limited.
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

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/functions/NEReorderLayer.h"
#include "arm_compute/runtime/NEON/functions/NETranspose.h"

#include "src/core/helpers/MemoryHelpers.h"
#include "src/cpu/kernels/assembly/arm_common/internal/utils.hpp"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/SimpleTensor.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/GEMM.h"

#include <array>

#ifndef BARE_METAL
#include <thread>
#endif

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          typename T,
          typename WEI_T = float,
          typename DST_T = float,
          typename REF_T = float>
class CpuGemmAssemblyDispatchGenericValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         shape_c,
               TensorShape         output_shape,
               float               alpha,
               float               beta,
               DataType            a_data_type,
               DataType            b_data_type,
               DataType            dst_data_type,
               bool                accumulate,
               bool                pretranspose_b,
               ActivationLayerInfo act_info,
               bool                fast_math    = false,
               bool                use_fp32_acc = false)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            !cpu_supports_dtypes({a_data_type, b_data_type, dst_data_type}))
        {
            return;
        }
        if (fast_math && !CPUInfo::get().has_bf16())
        {
            return;
        }
        ARM_COMPUTE_UNUSED(alpha);
        ARM_COMPUTE_UNUSED(beta);

        compute_target(shape_a, shape_b, shape_c, output_shape, a_data_type, b_data_type, dst_data_type, accumulate,
                       pretranspose_b, act_info, fast_math, use_fp32_acc);
        compute_reference(shape_a, shape_b, output_shape, dst_data_type, accumulate, act_info, fast_math);
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
            case DataType::BFLOAT16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<bfloat16> distribution{float(lo), float(hi),
                                                                                           true /* portable */};
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

    void compute_target(const TensorShape  &shape_a,
                        const TensorShape  &shape_b,
                        const TensorShape  &shape_c,
                        const TensorShape  &output_shape,
                        DataType            a_data_type,
                        DataType            b_data_type,
                        DataType            dst_data_type,
                        bool                accumulate,
                        bool                pretranspose_b,
                        ActivationLayerInfo act_info,
                        bool                fast_math,
                        bool                use_fp32_acc)
    {
        ARM_COMPUTE_UNUSED(shape_c);
        // Create tensors
        TensorType  a            = create_tensor<TensorType>(shape_a, a_data_type, 1);
        TensorType  b            = create_tensor<TensorType>(shape_b, b_data_type, 1);
        TensorType  b_transposed = create_tensor<TensorType>({shape_b[1], shape_b[0]}, b_data_type, 1);
        TensorType  dst          = create_tensor<TensorType>(output_shape, dst_data_type, 1);
        TensorType *c            = nullptr;
        // Create and configure function
        FunctionType gemm;
        NETranspose  transpose;

        add_padding_x({&a, &b, &b_transposed, &dst});

        GEMMInfo gemm_info;
        gemm_info.set_use_fp32_acc(use_fp32_acc);
        gemm_info.set_accumulate(accumulate);
        gemm_info.set_pretranspose_B(pretranspose_b);
        gemm_info.set_activation_info(act_info);
        gemm_info.set_fast_math(fast_math);

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

        _target = std::move(dst);
    }

    void compute_reference(const TensorShape  &shape_a,
                           const TensorShape  &shape_b,
                           const TensorShape  &output_shape,
                           DataType            data_type,
                           bool                accumulate,
                           ActivationLayerInfo act_info,
                           bool                fast_math = false)
    {
        // Create reference
        SimpleTensor<REF_T> a{shape_a, data_type, 1};
        SimpleTensor<REF_T> b{shape_b, data_type, 1};
        SimpleTensor<REF_T> c{output_shape, data_type, 1};
        SimpleTensor<REF_T> dst{output_shape, data_type, 1};

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
            reference::gemm_accumulate<REF_T>(a, b, c, 1.0f, 0.f, dst);
        }
        else
        {
            dst = reference::gemm<REF_T>(a, b, c, 1.f, 0.f, fast_math);
        }

        if (act_info.enabled())
        {
            dst = reference::activation_layer<REF_T>(dst, act_info);
        }

        _reference = dst;
    }

    TensorType          _target{};
    SimpleTensor<REF_T> _reference{};
};

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          typename T,
          typename WEI_T = T,
          typename DST_T = T,
          typename REF_T = T>
class CpuGemmAssemblyDispatchValidationFixture : protected CpuGemmAssemblyDispatchGenericValidationFixture<TensorType,
                                                                                                           AccessorType,
                                                                                                           FunctionType,
                                                                                                           T,
                                                                                                           WEI_T,
                                                                                                           DST_T,
                                                                                                           REF_T>
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         shape_c,
               TensorShape         output_shape,
               float               alpha,
               float               beta,
               DataType            a_data_type,
               DataType            b_data_type,
               DataType            dst_data_type,
               bool                accumulate,
               bool                pretranspose_b,
               ActivationLayerInfo act_info,
               bool                fast_math)
    {
        CpuGemmAssemblyDispatchGenericValidationFixture<TensorType, AccessorType, FunctionType, T, WEI_T, DST_T,
                                                        REF_T>::setup(shape_a, shape_b, shape_c, output_shape, alpha,
                                                                      beta, a_data_type, b_data_type, dst_data_type,
                                                                      accumulate, pretranspose_b, act_info, fast_math,
                                                                      false);
    }
};

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          typename T,
          typename WEI_T = T,
          typename DST_T = T,
          typename REF_T = T>
class CpuGemmAccF32AssemblyDispatchValidationFixture
    : protected CpuGemmAssemblyDispatchGenericValidationFixture<TensorType,
                                                                AccessorType,
                                                                FunctionType,
                                                                T,
                                                                WEI_T,
                                                                DST_T,
                                                                REF_T>
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
               bool                use_fp32_acc,
               ActivationLayerInfo act_info)
    {
        CpuGemmAssemblyDispatchGenericValidationFixture<TensorType, AccessorType, FunctionType, T, WEI_T, DST_T,
                                                        REF_T>::setup(shape_a, shape_b, shape_c, output_shape, alpha,
                                                                      beta, data_type, data_type, data_type, accumulate,
                                                                      pretranspose_b, act_info, false, use_fp32_acc);
    }
};

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          typename T,
          typename WEI_T = T,
          typename DST_T = float,
          typename REF_T = float>
class CpuGemmDstF32AssemblyDispatchValidationFixture
    : protected CpuGemmAssemblyDispatchGenericValidationFixture<TensorType,
                                                                AccessorType,
                                                                FunctionType,
                                                                T,
                                                                WEI_T,
                                                                DST_T,
                                                                REF_T>
{
public:
    void setup(TensorShape         shape_a,
               TensorShape         shape_b,
               TensorShape         shape_c,
               TensorShape         output_shape,
               float               alpha,
               float               beta,
               DataType            data_type,
               bool                pretranspose_b,
               ActivationLayerInfo act_info)
    {
        if ((std::is_same<TensorType, Tensor>::value && // Cpu
             data_type == DataType::F16 && !CPUInfo::get().has_fp16()) ||
            !CPUInfo::get().has_fhm())
        {
            return;
        }
        ARM_COMPUTE_UNUSED(alpha);
        ARM_COMPUTE_UNUSED(beta);
        this->compute_target(shape_a, shape_b, shape_c, output_shape, data_type, data_type, DataType::F32, false,
                             pretranspose_b, act_info, false, false);
        compute_reference(shape_a, shape_b, output_shape, data_type, act_info);
    }

protected:
    void compute_reference(const TensorShape  &shape_a,
                           const TensorShape  &shape_b,
                           const TensorShape  &output_shape,
                           DataType            data_type,
                           ActivationLayerInfo act_info)
    {
        // Create reference
        SimpleTensor<T>     a{shape_a, data_type, 1};
        SimpleTensor<T>     b{shape_b, data_type, 1};
        SimpleTensor<T>     c{output_shape, data_type, 1};
        SimpleTensor<float> dst{output_shape, DataType::F32, 1};

        // Fill reference
        this->fill(a, 0, -1.f, 1.f);
        this->fill(b, 1, -1.f, 1.f);
        this->fill(c, 2);

        dst = reference::gemm_mixed_precision<T, float>(a, b, c, 1.f, 0.f);

        if (act_info.enabled())
        {
            dst = reference::activation_layer<float>(dst, act_info);
        }
        this->_reference = dst;
    }
};

#ifdef ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS
// The NumRuns template parameter supports thread-safety tests by creating an
// array at compile-time to store results from parallely runs.
//
// In general, we should avoid adding further parameters where possible as it
// bloats the validation binary. Please consider if your use-case can be met
// without expanding the template.
template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          typename T,
          typename WEI_T = T,
          typename DST_T = T,
          typename REF_T = T,
          int NumRuns    = 1>
class CpuGemmAssemblyDispatchFixedFormatGenericValidationFixture
    : public CpuGemmAssemblyDispatchGenericValidationFixture<TensorType,
                                                             AccessorType,
                                                             FunctionType,
                                                             T,
                                                             WEI_T,
                                                             DST_T,
                                                             REF_T>
{
public:
    void setup(TensorShape shape_a,
               TensorShape shape_b,
               TensorShape shape_c,
               TensorShape output_shape,
               float       alpha,
               float       beta,
               DataType    a_data_type,
               DataType    b_data_type,
               DataType    dst_data_type,
               TestType    test_type)
    {
        ARM_COMPUTE_UNUSED(alpha);
        ARM_COMPUTE_UNUSED(beta);
        bool fast_math = a_data_type == DataType::BFLOAT16 || b_data_type == DataType::BFLOAT16 ||
                         dst_data_type == DataType::BFLOAT16;
        _test_type = test_type;
        compute_target(shape_a, shape_b, shape_c, output_shape, a_data_type, b_data_type,
                       fast_math ? DataType::F32 : dst_data_type);
        compute_reference(shape_a, shape_b, output_shape, fast_math ? DataType::F32 : dst_data_type, false,
                          ActivationLayerInfo());
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

        const TensorShape TS(
            {tensor_shape[0], arm_compute::ceil_to_multiple<int32_t, int32_t>(tensor_shape[1], interleave_by)});

        TensorInfo new_tensor_info = tensor_info;
        new_tensor_info.set_data_layout(DataLayout::UNKNOWN);
        new_tensor_info.init(TS, tensor_info.num_channels(), data_type, strides_in_bytes, offset_first_element_in_bytes,
                             total_size_in_bytes);
        return new_tensor_info;
    }

    void compute_target(TensorShape shape_a,
                        TensorShape shape_b,
                        TensorShape shape_c,
                        TensorShape output_shape,
                        DataType    a_data_type,
                        DataType    b_data_type,
                        DataType    dst_data_type)
    {
        ARM_COMPUTE_UNUSED(shape_c);
        permute(shape_b, PermutationVector(1U, 0U));
        // Create tensors
        std::array<TensorType, NumRuns> a{};
        std::array<TensorType, NumRuns> b{};
        std::array<TensorType, NumRuns> c{};
        std::array<TensorType, NumRuns> dst{};
        std::array<TensorType, NumRuns> b_transformed{};

        // Create and configure function
        FunctionType              gemm;
        NEReorderLayer            reorder;
        arm_compute::WeightFormat computed_weight_format{arm_compute::WeightFormat::ANY};
        GEMMInfo                  gemm_info;

        for (int i = 0; i < NumRuns; ++i)
        {
            a[i]   = create_tensor<TensorType>(shape_a, a_data_type, 1, QuantizationInfo(), DataLayout::NCHW);
            b[i]   = create_tensor<TensorType>(shape_b, b_data_type, 1, QuantizationInfo(), DataLayout::NCHW);
            dst[i] = create_tensor<TensorType>(output_shape, dst_data_type, 1, QuantizationInfo(), DataLayout::NCHW);
            ARM_COMPUTE_ASSERT(a[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(b[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(dst[i].info()->is_resizable());
        }

        gemm_info.set_fixed_format(true);
        gemm_info.set_accumulate(false);
        gemm_info.set_weight_format(computed_weight_format);

        const bool kernel_found = bool(FunctionType::has_opt_impl(computed_weight_format, a[0].info(), b[0].info(),
                                                                  nullptr, dst[0].info(), gemm_info));

        ARM_COMPUTE_ASSERT(kernel_found);
        gemm_info.set_weight_format(computed_weight_format);
        gemm_info.set_fast_math(is_fixed_format_fast_math(computed_weight_format));

        // Allocate tensors
        for (int i = 0; i < NumRuns; ++i)
        {
            b_transformed[i] = create_tensor<TensorType>(prepare_weights(*b[i].info(), computed_weight_format));
            ARM_COMPUTE_ASSERT(b_transformed[i].info()->is_resizable());

            a[i].info()->set_are_values_constant(false);
            b_transformed[i].info()->set_are_values_constant(false);

            a[i].allocator()->allocate();
            b[i].allocator()->allocate();
            b_transformed[i].allocator()->allocate();
            dst[i].allocator()->allocate();

            ARM_COMPUTE_ASSERT(!a[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!b[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!b_transformed[i].info()->is_resizable());
            ARM_COMPUTE_ASSERT(!dst[i].info()->is_resizable());

            // Fill tensors
            this->fill(AccessorType(a[i]), 7 * i + 0, -1.f, 1.f);
            this->fill(AccessorType(b[i]), 7 * i + 1, -1.f, 1.f);

            // Reorder weight to the expected format
            ARM_COMPUTE_ASSERT(reorder.validate(b[i].info(), b_transformed[i].info(), WeightFormat::OHWI,
                                                computed_weight_format, true));
            reorder.configure(&b[i], &b_transformed[i], WeightFormat::OHWI, computed_weight_format, true);
            reorder.run();
        }

        ARM_COMPUTE_ASSERT(gemm.validate(a[0].info(), b_transformed[0].info(), nullptr, dst[0].info(), gemm_info));
        gemm.configure(a[0].info(), b_transformed[0].info(), nullptr, dst[0].info(), gemm_info);
        ARM_COMPUTE_ASSERT(gemm.is_configured());

        std::array<ITensorPack, NumRuns> run_pack;
        std::array<ITensorPack, NumRuns> prep_pack;

        // Define this as a lambda so we can call it in a thread or within the
        // main process (if BARE_METAL)
        auto gemm_runner = [&](int i)
        {
            experimental::MemoryRequirements aux_mem_req = gemm.workspace();
            MemoryGroup                      memory_group{};

            WorkspaceData<Tensor> workspace =
                manage_workspace<Tensor>(aux_mem_req, memory_group, run_pack[i], prep_pack[i]);

            gemm.prepare(prep_pack[i]);
            MemoryGroupResourceScope scope_mg(memory_group);

            auto has_reshape =
                std::find_if(aux_mem_req.begin(), aux_mem_req.end(),
                             [](const arm_compute::experimental::MemoryInfo &m) -> bool
                             { return m.lifetime == arm_compute::experimental::MemoryLifetime::Persistent; });

            if (has_reshape != std::end(aux_mem_req))
            {
                b_transformed[i].mark_as_unused();
            }
            else
            {
                run_pack[i].add_const_tensor(ACL_SRC_1, &b_transformed[i]);
            }

            // Release temporary tensors that are only used in prepare stage
            release_temporaries<Tensor>(aux_mem_req, workspace);

            // Run, and store result
            gemm.run(run_pack[i]);
            _target[i] = std::move(dst[i]);
        };

#ifndef BARE_METAL
        std::array<std::thread, NumRuns> threads;
#endif // BARE_METAL

        for (int i = 0; i < NumRuns; ++i)
        {
            run_pack[i].add_const_tensor(arm_compute::TensorType::ACL_SRC_0, &a[i]);
            run_pack[i].add_const_tensor(arm_compute::TensorType::ACL_SRC_1, &b_transformed[i]);
            run_pack[i].add_tensor(arm_compute::TensorType::ACL_SRC_2, &c[i]);
            run_pack[i].add_tensor(arm_compute::TensorType::ACL_DST, &dst[i]);

            // Prepare memory
            prep_pack[i] = {{arm_compute::TensorType::ACL_SRC_1, &b_transformed[i]},
                            {arm_compute::TensorType::ACL_SRC_2, &c[i]}};

            if (_test_type == TestType::ConfigureOnceRunMultiThreaded)
            {
#ifndef BARE_METAL
                threads[i] = std::thread{gemm_runner, i};
#endif // BARE_METAL
            }
            else
            {
                gemm_runner(i);
            }
        }

        for (int i = 0; i < NumRuns; ++i)
        {
#ifndef BARE_METAL
            if (_test_type == TestType::ConfigureOnceRunMultiThreaded)
            {
                threads[i].join();
            }
#endif // BARE_METAL

            a[i].allocator()->free();
            b[i].allocator()->free();
            b_transformed[i].allocator()->free();
        }
    }

    void compute_reference(const TensorShape  &shape_a,
                           const TensorShape  &shape_b,
                           const TensorShape  &output_shape,
                           DataType            data_type,
                           bool                accumulate,
                           ActivationLayerInfo act_info,
                           bool                fast_math = false)
    {
        // Create reference
        SimpleTensor<REF_T> a{shape_a, data_type, 1};
        SimpleTensor<REF_T> b{shape_b, data_type, 1};
        SimpleTensor<REF_T> c{output_shape, data_type, 1};
        SimpleTensor<REF_T> dst{output_shape, data_type, 1};

        for (int i = 0; i < NumRuns; ++i)
        {
            // Fill reference
            this->fill(a, 7 * i + 0, -1.f, 1.f);
            this->fill(b, 7 * i + 1, -1.f, 1.f);
            this->fill(c, 7 * i + 2);

            // Do in place summation
            if (accumulate)
            {
                this->fill(dst, 7 * i + 6);
            }

            if (accumulate)
            {
                reference::gemm_accumulate<REF_T>(a, b, c, 1.0f, 0.f, dst);
            }
            else
            {
                dst = reference::gemm<REF_T>(a, b, c, 1.f, 0.f, fast_math);
            }

            if (act_info.enabled())
            {
                dst = reference::activation_layer<REF_T>(dst, act_info);
            }

            _reference[i] = dst;
        }
    }

    std::array<TensorType, NumRuns>          _target{};
    std::array<SimpleTensor<REF_T>, NumRuns> _reference{};
    TestType                                 _test_type{TestType::ConfigureOnceRunOnce};
};

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          typename T,
          typename WEI_T = T,
          typename DST_T = T,
          typename REF_T = T>
class CpuGemmAssemblyDispatchFixedFormatFixture
    : public CpuGemmAssemblyDispatchFixedFormatGenericValidationFixture<TensorType,
                                                                        AccessorType,
                                                                        FunctionType,
                                                                        T,
                                                                        WEI_T,
                                                                        DST_T,
                                                                        REF_T>
{
public:
    void setup(TensorShape shape_a,
               TensorShape shape_b,
               TensorShape shape_c,
               TensorShape output_shape,
               float       alpha,
               float       beta,
               DataType    src_data_type,
               DataType    wei_data_type,
               DataType    dst_data_type)
    {
        CpuGemmAssemblyDispatchFixedFormatGenericValidationFixture<TensorType, AccessorType, FunctionType, T, WEI_T,
                                                                   DST_T, REF_T>::setup(shape_a, shape_b, shape_c,
                                                                                        output_shape, alpha, beta,
                                                                                        src_data_type, wei_data_type,
                                                                                        dst_data_type,
                                                                                        TestType::ConfigureOnceRunOnce);
    }
};

template <typename TensorType,
          typename AccessorType,
          typename FunctionType,
          typename T,
          typename WEI_T = T,
          typename DST_T = T,
          typename REF_T = T,
          int NumRuns    = 1>
class CpuGemmAssemblyDispatchFixedFormatThreadSafetyFixture
    : public CpuGemmAssemblyDispatchFixedFormatGenericValidationFixture<TensorType,
                                                                        AccessorType,
                                                                        FunctionType,
                                                                        T,
                                                                        WEI_T,
                                                                        DST_T,
                                                                        REF_T,
                                                                        NumRuns>
{
public:
    void setup(TensorShape shape_a,
               TensorShape shape_b,
               TensorShape shape_c,
               TensorShape output_shape,
               float       alpha,
               float       beta,
               DataType    a_data_type,
               DataType    b_data_type,
               DataType    dst_data_type)
    {
        CpuGemmAssemblyDispatchFixedFormatGenericValidationFixture<
            TensorType, AccessorType, FunctionType, T, WEI_T, DST_T, REF_T,
            NumRuns>::setup(shape_a, shape_b, shape_c, output_shape, alpha, beta, a_data_type, b_data_type,
                            dst_data_type, TestType::ConfigureOnceRunMultiThreaded);
    }
};

#endif //ARM_COMPUTE_ENABLE_FIXED_FORMAT_KERNELS

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUGEMMASSEMBLYDISPATCHFIXTURE_H
