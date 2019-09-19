/*
 * Copyright (c) 2019 ARM Limited.
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
#include "tests/validation/reference/NormalizationLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename AllocatorType,
          typename LifetimeMgrType,
          typename PoolMgrType,
          typename MemoryMgrType>
struct MemoryManagementService
{
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
        ARM_COMPUTE_EXPECT(mm->pool_manager() != nullptr, framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(mm->lifetime_manager() != nullptr, framework::LogLevel::ERRORS);

        if(validate_finalized)
        {
            ARM_COMPUTE_EXPECT(mm->lifetime_manager()->are_all_finalized(), framework::LogLevel::ERRORS);
        }
        ARM_COMPUTE_EXPECT(mm->pool_manager()->num_pools() == num_pools, framework::LogLevel::ERRORS);
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
    }
    void run()
    {
        _func.run();
    }

private:
    FuncType _func;
};
} // namespace

/** Simple test case to run a single function with different shapes twice.
 *
 * Runs a specified function twice, where the second time the size of the input/output is different
 * Internal memory of the function and input/output are managed by different services
 */
template <typename TensorType,
          typename AccessorType,
          typename AllocatorType,
          typename LifetimeMgrType,
          typename PoolMgrType,
          typename MemoryManagerType,
          typename SimpleFunctionWrapperType>
class DynamicTensorType3SingleFunction : public framework::Fixture
{
    using T                           = float;
    using MemoryManagementServiceType = MemoryManagementService<AllocatorType, LifetimeMgrType, PoolMgrType, MemoryManagerType>;

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

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

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
    TensorShape                         input_l0{}, input_l1{};
    typename LifetimeMgrType::info_type internal_l0{}, internal_l1{};
    typename LifetimeMgrType::info_type cross_l0{}, cross_l1{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_DYNAMIC_TENSOR */
