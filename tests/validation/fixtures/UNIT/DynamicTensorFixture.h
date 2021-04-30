/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_UNIT_DYNAMIC_TENSOR
#define ARM_COMPUTE_TEST_UNIT_DYNAMIC_TENSOR

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/NormalizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename AllocatorType,
          typename LifetimeMgrType,
          typename PoolMgrType,
          typename MemoryMgrType>
struct MemoryManagementService
{
public:
    using LftMgrType = LifetimeMgrType;

public:
    MemoryManagementService()
        : allocator(), lifetime_mgr(nullptr), pool_mgr(nullptr), mm(nullptr), mg(), num_pools(0)
    {
        lifetime_mgr = std::make_shared<LifetimeMgrType>();
        pool_mgr     = std::make_shared<PoolMgrType>();
        mm           = std::make_shared<MemoryMgrType>(lifetime_mgr, pool_mgr);
        mg           = MemoryGroup(mm);
    }

    void populate(size_t pools)
    {
        mm->populate(allocator, pools);
        num_pools = pools;
    }

    void clear()
    {
        mm->clear();
        num_pools = 0;
    }

    void validate(bool validate_finalized) const
    {
        ARM_COMPUTE_ASSERT(mm->pool_manager() != nullptr);
        ARM_COMPUTE_ASSERT(mm->lifetime_manager() != nullptr);

        if(validate_finalized)
        {
            ARM_COMPUTE_ASSERT(mm->lifetime_manager()->are_all_finalized());
        }
        ARM_COMPUTE_ASSERT(mm->pool_manager()->num_pools() == num_pools);
    }

    AllocatorType                    allocator;
    std::shared_ptr<LifetimeMgrType> lifetime_mgr;
    std::shared_ptr<PoolMgrType>     pool_mgr;
    std::shared_ptr<MemoryMgrType>   mm;
    MemoryGroup                      mg;
    size_t                           num_pools;
};

template <typename MemoryMgrType, typename FuncType, typename ITensorType>
class SimpleFunctionWrapper
{
public:
    SimpleFunctionWrapper(std::shared_ptr<MemoryMgrType> mm)
        : _func(mm)
    {
    }
    void configure(ITensorType *src, ITensorType *dst)
    {
        ARM_COMPUTE_UNUSED(src, dst);
    }
    void run()
    {
        _func.run();
    }

private:
    FuncType _func;
};

/** Simple test case to run a single function with different shapes twice.
 *
 * Runs a specified function twice, where the second time the size of the input/output is different
 * Internal memory of the function and input/output are managed by different services
 */
template <typename TensorType,
          typename AccessorType,
          typename MemoryManagementServiceType,
          typename SimpleFunctionWrapperType>
class DynamicTensorType3SingleFunction : public framework::Fixture
{
    using T = float;

public:
    template <typename...>
    void setup(TensorShape input_level0, TensorShape input_level1)
    {
        input_l0 = input_level0;
        input_l1 = input_level1;
        run();
    }

protected:
    void run()
    {
        MemoryManagementServiceType serv_internal;
        MemoryManagementServiceType serv_cross;
        const size_t                num_pools          = 1;
        const bool                  validate_finalized = true;

        // Create Tensor shapes.
        TensorShape level_0 = TensorShape(input_l0);
        TensorShape level_1 = TensorShape(input_l1);

        // Level 0
        // Create tensors
        TensorType src = create_tensor<TensorType>(level_0, DataType::F32, 1);
        TensorType dst = create_tensor<TensorType>(level_0, DataType::F32, 1);

        serv_cross.mg.manage(&src);
        serv_cross.mg.manage(&dst);

        // Create and configure function
        SimpleFunctionWrapperType layer(serv_internal.mm);
        layer.configure(&src, &dst);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Populate and validate memory manager
        serv_cross.populate(num_pools);
        serv_internal.populate(num_pools);
        serv_cross.validate(validate_finalized);
        serv_internal.validate(validate_finalized);

        // Extract lifetime manager meta-data information
        internal_l0 = serv_internal.lifetime_mgr->info();
        cross_l0    = serv_cross.lifetime_mgr->info();

        // Acquire memory manager, fill tensors and compute functions
        serv_cross.mg.acquire();
        arm_compute::test::library->fill_tensor_value(AccessorType(src), 12.f);
        layer.run();
        serv_cross.mg.release();

        // Clear manager
        serv_cross.clear();
        serv_internal.clear();
        serv_cross.validate(validate_finalized);
        serv_internal.validate(validate_finalized);

        // Level 1
        // Update the tensor shapes
        src.info()->set_tensor_shape(level_1);
        dst.info()->set_tensor_shape(level_1);
        src.info()->set_is_resizable(true);
        dst.info()->set_is_resizable(true);

        serv_cross.mg.manage(&src);
        serv_cross.mg.manage(&dst);

        // Re-configure the function
        layer.configure(&src, &dst);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        // Populate and validate memory manager
        serv_cross.populate(num_pools);
        serv_internal.populate(num_pools);
        serv_cross.validate(validate_finalized);
        serv_internal.validate(validate_finalized);

        // Extract lifetime manager meta-data information
        internal_l1 = serv_internal.lifetime_mgr->info();
        cross_l1    = serv_cross.lifetime_mgr->info();

        // Compute functions
        serv_cross.mg.acquire();
        arm_compute::test::library->fill_tensor_value(AccessorType(src), 12.f);
        layer.run();
        serv_cross.mg.release();

        // Clear manager
        serv_cross.clear();
        serv_internal.clear();
        serv_cross.validate(validate_finalized);
        serv_internal.validate(validate_finalized);
    }

public:
    TensorShape                                                 input_l0{}, input_l1{};
    typename MemoryManagementServiceType::LftMgrType::info_type internal_l0{}, internal_l1{};
    typename MemoryManagementServiceType::LftMgrType::info_type cross_l0{}, cross_l1{};
};

/** Simple test case to run a single function with different shapes twice.
 *
 * Runs a specified function twice, where the second time the size of the input/output is different
 * Internal memory of the function and input/output are managed by different services
 */
template <typename TensorType,
          typename AccessorType,
          typename MemoryManagementServiceType,
          typename ComplexFunctionType>
class DynamicTensorType3ComplexFunction : public framework::Fixture
{
    using T = float;

public:
    template <typename...>
    void setup(std::vector<TensorShape> input_shapes, TensorShape weights_shape, TensorShape bias_shape, std::vector<TensorShape> output_shapes, PadStrideInfo info)
    {
        num_iterations = input_shapes.size();
        _data_type     = DataType::F32;
        _data_layout   = DataLayout::NHWC;
        _input_shapes  = input_shapes;
        _output_shapes = output_shapes;
        _weights_shape = weights_shape;
        _bias_shape    = bias_shape;
        _info          = info;

        // Create function
        _f_target = std::make_unique<ComplexFunctionType>(_ms.mm);
    }

    void run_iteration(unsigned int idx)
    {
        auto input_shape  = _input_shapes[idx];
        auto output_shape = _output_shapes[idx];

        dst_ref    = run_reference(input_shape, _weights_shape, _bias_shape, output_shape, _info);
        dst_target = run_target(input_shape, _weights_shape, _bias_shape, output_shape, _info, WeightsInfo());
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType run_target(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape,
                          PadStrideInfo info, WeightsInfo weights_info)
    {
        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        _weights_target = create_tensor<TensorType>(weights_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        _bias_target    = create_tensor<TensorType>(bias_shape, _data_type, 1);

        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType dst = create_tensor<TensorType>(output_shape, _data_type, 1, QuantizationInfo(), _data_layout);

        // Create and configure function
        _f_target->configure(&src, &_weights_target, &_bias_target, &dst, info, weights_info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
        _weights_target.allocator()->allocate();
        _bias_target.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(_weights_target), 1);
        fill(AccessorType(_bias_target), 2);

        // Populate and validate memory manager
        _ms.clear();
        _ms.populate(1);
        _ms.mg.acquire();

        // Compute NEConvolutionLayer function
        _f_target->run();
        _ms.mg.release();

        return dst;
    }

    SimpleTensor<T> run_reference(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, _data_type, 1 };
        SimpleTensor<T> weights{ weights_shape, _data_type, 1 };
        SimpleTensor<T> bias{ bias_shape, _data_type, 1 };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);
        fill(bias, 2);

        return reference::convolution_layer<T>(src, weights, bias, output_shape, info);
    }

public:
    unsigned int    num_iterations{ 0 };
    SimpleTensor<T> dst_ref{};
    TensorType      dst_target{};

private:
    DataType                             _data_type{ DataType::UNKNOWN };
    DataLayout                           _data_layout{ DataLayout::UNKNOWN };
    PadStrideInfo                        _info{};
    std::vector<TensorShape>             _input_shapes{};
    std::vector<TensorShape>             _output_shapes{};
    TensorShape                          _weights_shape{};
    TensorShape                          _bias_shape{};
    MemoryManagementServiceType          _ms{};
    TensorType                           _weights_target{};
    TensorType                           _bias_target{};
    std::unique_ptr<ComplexFunctionType> _f_target{};
};

/** Fixture that create a pipeline of Convolutions and changes the inputs dynamically
 *
 * Runs a list of convolutions and then resizes the inputs and reruns.
 * Updates the memory manager and allocated memory.
 */
template <typename TensorType,
          typename AccessorType,
          typename MemoryManagementServiceType,
          typename ComplexFunctionType>
class DynamicTensorType2PipelineFunction : public framework::Fixture
{
    using T = float;

public:
    template <typename...>
    void setup(std::vector<TensorShape> input_shapes)
    {
        _data_type    = DataType::F32;
        _data_layout  = DataLayout::NHWC;
        _input_shapes = input_shapes;

        run();
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    void run()
    {
        const unsigned int num_functions = 5;
        const unsigned int num_tensors   = num_functions + 1;
        const unsigned int num_resizes   = _input_shapes.size();

        for(unsigned int i = 0; i < num_functions; ++i)
        {
            _functions.emplace_back(std::make_unique<ComplexFunctionType>(_ms.mm));
        }

        for(unsigned int i = 0; i < num_resizes; ++i)
        {
            TensorShape   input_shape   = _input_shapes[i];
            TensorShape   weights_shape = TensorShape(3U, 3U, input_shape[2], input_shape[2]);
            TensorShape   output_shape  = input_shape;
            PadStrideInfo info(1U, 1U, 1U, 1U);

            if(_data_layout == DataLayout::NHWC)
            {
                permute(input_shape, PermutationVector(2U, 0U, 1U));
                permute(weights_shape, PermutationVector(2U, 0U, 1U));
                permute(output_shape, PermutationVector(2U, 0U, 1U));
            }

            std::vector<TensorType> tensors(num_tensors);
            std::vector<TensorType> ws(num_functions);
            std::vector<TensorType> bs(num_functions);

            auto tensor_info  = TensorInfo(input_shape, 1, _data_type);
            auto weights_info = TensorInfo(weights_shape, 1, _data_type);
            tensor_info.set_data_layout(_data_layout);
            weights_info.set_data_layout(_data_layout);

            tensors[0].allocator()->init(tensor_info);
            for(unsigned int f = 0; f < num_functions; ++f)
            {
                tensors[f + 1].allocator()->init(tensor_info);
                ws[f].allocator()->init(weights_info);

                _functions[f]->configure(&tensors[f], &ws[f], nullptr, &tensors[f + 1], info);

                // Allocate tensors
                tensors[f].allocator()->allocate();
                ws[f].allocator()->allocate();
            }
            tensors[num_functions].allocator()->allocate();

            // Populate and validate memory manager
            _ms.clear();
            _ms.populate(1);
            _ms.mg.acquire();

            // Run pipeline
            for(unsigned int f = 0; f < num_functions; ++f)
            {
                _functions[f]->run();
            }

            // Release memory group
            _ms.mg.release();
        }
    }

private:
    DataType                                          _data_type{ DataType::UNKNOWN };
    DataLayout                                        _data_layout{ DataLayout::UNKNOWN };
    std::vector<TensorShape>                          _input_shapes{};
    MemoryManagementServiceType                       _ms{};
    std::vector<std::unique_ptr<ComplexFunctionType>> _functions{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_DYNAMIC_TENSOR */
