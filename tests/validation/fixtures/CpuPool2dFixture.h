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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_CPUPOOL2DFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_CPUPOOL2DFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"  // required for PoolingLayerInfo
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"

#include "tests/AssetsLibrary.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/Globals.h"
#include "tests/validation/reference/PoolingLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuPool2dValidationGenericFixture : public framework::Fixture
{
public:
    void setup(TensorShape      shape,
               PoolingLayerInfo pool_info,
               DataType         data_type,
               DataLayout       data_layout,
               bool             indices      = false,
               QuantizationInfo input_qinfo  = QuantizationInfo(),
               QuantizationInfo output_qinfo = QuantizationInfo(),
               bool             mixed_layout = false)
    {
        if (std::is_same<TensorType, Tensor>::value && // Cpu
            data_type == DataType::F16 && !CPUInfo::get().has_fp16())
        {
            return;
        }

        _mixed_layout = mixed_layout;
        _pool_info    = pool_info;
        _target       = compute_target(shape, pool_info, data_type, data_layout, input_qinfo, output_qinfo, indices);
        _reference    = compute_reference(shape, pool_info, data_type, data_layout, input_qinfo, output_qinfo, indices);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if (tensor.data_type() == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, 0);
        }
        else if (tensor.data_type() == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution{-1.0f, 1.0f};
            library->fill(tensor, distribution, 0);
        }
        else // data type is quantized_asymmetric
        {
            library->fill_tensor_uniform(tensor, 0);
        }
    }

    TensorType compute_target(TensorShape      shape,
                              PoolingLayerInfo info,
                              DataType         data_type,
                              DataLayout       data_layout,
                              QuantizationInfo input_qinfo,
                              QuantizationInfo output_qinfo,
                              bool             indices)
    {
        // Change shape in case of NHWC.
        if (data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }
        // Create tensors
        TensorType        src       = create_tensor<TensorType>(shape, data_type, 1, input_qinfo, data_layout);
        const TensorShape dst_shape = misc::shape_calculator::compute_pool_shape(*(src.info()), info);
        TensorType        dst       = create_tensor<TensorType>(dst_shape, data_type, 1, output_qinfo, data_layout);
        _target_indices             = create_tensor<TensorType>(dst_shape, DataType::U32, 1, output_qinfo, data_layout);

        // Create and configure function
        FunctionType pooling;
        pooling.configure(src.info(), dst.info(), info, (indices) ? _target_indices.info() : nullptr);
        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());
        ARM_COMPUTE_ASSERT(_target_indices.info()->is_resizable());

        add_padding_x({&src, &dst, &_target_indices}, data_layout);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        _target_indices.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!_target_indices.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));

        ITensorPack run_pack;
        if (indices)
        {
            run_pack.add_tensor(arm_compute::TensorType::ACL_SRC_0, &src);
            run_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst);
            run_pack.add_tensor(arm_compute::TensorType::ACL_DST_1, &_target_indices);
        }
        else
        {
            run_pack.add_tensor(arm_compute::TensorType::ACL_SRC_0, &src);
            run_pack.add_tensor(arm_compute::TensorType::ACL_DST, &dst);
        }

        auto mg = MemoryGroup{};
        auto ws = manage_workspace<Tensor>(pooling.workspace(), mg, run_pack);

        if (_mixed_layout)
        {
            //added for mixed layout
            const DataLayout data_layout = src.info()->data_layout();

            // Test Multi DataLayout graph cases, when the data layout changes after configure
            src.info()->set_data_layout(data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
            dst.info()->set_data_layout(data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

            // Compute function
            pooling.run(run_pack);
        }
        else
        {
            // Compute function
            pooling.run(run_pack);
        }
        return dst;
    }

    SimpleTensor<T> compute_reference(TensorShape      shape,
                                      PoolingLayerInfo info,
                                      DataType         data_type,
                                      DataLayout       data_layout,
                                      QuantizationInfo input_qinfo,
                                      QuantizationInfo output_qinfo,
                                      bool             indices)
    {
        // Create reference
        SimpleTensor<T> src(shape, data_type, 1, input_qinfo);
        // Fill reference
        fill(src);
        return reference::pooling_layer<T>(src, info, output_qinfo, indices ? &_ref_indices : nullptr, data_layout);
    }

    TensorType             _target{};
    SimpleTensor<T>        _reference{};
    PoolingLayerInfo       _pool_info{};
    bool                   _mixed_layout{false};
    TensorType             _target_indices{};
    SimpleTensor<uint32_t> _ref_indices{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuPool2dIndicesValidationFixture
    : public CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape   shape,
               PoolingType   pool_type,
               Size2D        pool_size,
               PadStrideInfo pad_stride_info,
               bool          exclude_padding,
               DataType      data_type,
               DataLayout    data_layout,
               bool          use_kernel_indices)
    {
        CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape,
            PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding, false, true,
                             use_kernel_indices),
            data_type, data_layout, true);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class CpuPool2dValidationFixture : public CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape   shape,
               PoolingType   pool_type,
               Size2D        pool_size,
               PadStrideInfo pad_stride_info,
               bool          exclude_padding,
               DataType      data_type,
               DataLayout    data_layout)
    {
        CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding), data_type,
            data_layout, false, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class CpuPool2dValidationMixedPrecisionFixture
    : public CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape   shape,
               PoolingType   pool_type,
               Size2D        pool_size,
               PadStrideInfo pad_stride_info,
               bool          exclude_padding,
               DataType      data_type,
               DataLayout    data_layout,
               bool          fp_mixed_precision = false)
    {
        CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape,
            PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding, fp_mixed_precision),
            data_type, data_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class CpuPool2dValidationQuantizedFixture
    : public CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape      shape,
               PoolingType      pool_type,
               Size2D           pool_size,
               PadStrideInfo    pad_stride_info,
               bool             exclude_padding,
               DataType         data_type,
               DataLayout       data_layout  = DataLayout::NCHW,
               QuantizationInfo input_qinfo  = QuantizationInfo(),
               QuantizationInfo output_qinfo = QuantizationInfo())
    {
        CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, PoolingLayerInfo(pool_type, pool_size, data_layout, pad_stride_info, exclude_padding), data_type,
            data_layout, false, input_qinfo, output_qinfo, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SpecialCpuPool2dValidationFixture
    : public CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape src_shape, PoolingLayerInfo pool_info, DataType data_type)
    {
        CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            src_shape, pool_info, data_type, pool_info.data_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GlobalCpuPool2dValidationFixture
    : public CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape shape, PoolingType pool_type, DataType data_type, DataLayout data_layout = DataLayout::NCHW)
    {
        CpuPool2dValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(
            shape, PoolingLayerInfo(pool_type, data_layout), data_type, data_layout);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_CPUPOOL2DFIXTURE_H
