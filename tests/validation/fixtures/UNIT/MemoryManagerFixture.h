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
#ifndef ARM_COMPUTE_TEST_UNIT_MEMORY_MANAGER
#define ARM_COMPUTE_TEST_UNIT_MEMORY_MANAGER

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/FullyConnectedLayer.h"
#include "tests/validation/reference/SoftmaxLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
/** Simple test case to run two fully connected layers using a blob affinity memory manager
 *
 * Runs two fully connected layers back to back
 */
template <typename TensorType, typename AccessorType, typename AllocatorType, typename FullyConnectedFunction>
class BlobMemoryManagerSimpleTestCaseFixture : public framework::Fixture
{
    using T = float;

public:
    void setup()
    {
        _target    = compute_target();
        _reference = compute_reference();
    };

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(0.5f, 1.f);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target()
    {
        auto lifetime_mgr = std::make_shared<BlobLifetimeManager>();
        auto pool_mgr     = std::make_shared<PoolManager>();
        auto mm           = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

        // Create tensors
        TensorType w1  = create_tensor<TensorType>(TensorShape(128U, 128U), DataType::F32, 1);
        TensorType b1  = create_tensor<TensorType>(TensorShape(128U), DataType::F32, 1);
        TensorType w2  = create_tensor<TensorType>(TensorShape(128U, 24U), DataType::F32, 1);
        TensorType b2  = create_tensor<TensorType>(TensorShape(24U), DataType::F32, 1);
        TensorType src = create_tensor<TensorType>(TensorShape(128U), DataType::F32, 1);
        TensorType fc1 = create_tensor<TensorType>(TensorShape(128U), DataType::F32, 1);
        TensorType dst = create_tensor<TensorType>(TensorShape(24U), DataType::F32, 1);

        // Create and configure function
        FullyConnectedFunction fc_layer_1(mm);
        FullyConnectedFunction fc_layer_2(mm);
        fc_layer_1.configure(&src, &w1, &b1, &fc1);
        fc_layer_2.configure(&fc1, &w2, &b2, &dst);

        // Allocate tensors
        w1.allocator()->allocate();
        b1.allocator()->allocate();
        w2.allocator()->allocate();
        b2.allocator()->allocate();
        src.allocator()->allocate();
        fc1.allocator()->allocate();
        dst.allocator()->allocate();

        // Finalize memory manager
        mm->populate(_allocator, 1 /* num_pools */);
        ARM_COMPUTE_ASSERT(mm->lifetime_manager()->are_all_finalized());
        ARM_COMPUTE_ASSERT(mm->pool_manager()->num_pools() == 1);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(w1), 1);
        fill(AccessorType(b1), 2);
        fill(AccessorType(w2), 3);
        fill(AccessorType(b2), 4);

        // Compute functions
        fc_layer_1.run();
        fc_layer_2.run();

        return dst;
    }

    SimpleTensor<T> compute_reference()
    {
        // Create reference
        SimpleTensor<T> w1{ TensorShape(128U, 128U), DataType::F32 };
        SimpleTensor<T> b1{ TensorShape(128U), DataType::F32 };
        SimpleTensor<T> w2{ TensorShape(128U, 24U), DataType::F32 };
        SimpleTensor<T> b2{ TensorShape(24U), DataType::F32 };
        SimpleTensor<T> src{ TensorShape(128U), DataType::F32 };

        // Fill reference
        fill(src, 0);
        fill(w1, 1);
        fill(b1, 2);
        fill(w2, 3);
        fill(b2, 4);

        auto fc1 = reference::fully_connected_layer(src, w1, b1, TensorShape(128U));
        return reference::fully_connected_layer(fc1, w2, b2, TensorShape(24U));
    }

protected:
    TensorType      _target{};
    SimpleTensor<T> _reference{};
    AllocatorType   _allocator{};
};

/** Test case to run two fully connected layers using a blob affinity memory manager,
 *  reconfigure with different shapes and rerun
 *
 * Runs two fully connected layers back to back then reconfigures with different batch size and reruns
 * Shapes of the reconfigure step are smaller that the initial configured step
 */
template <typename TensorType, typename AccessorType, typename AllocatorType, typename FullyConnectedFunction>
class BlobMemoryManagerReconfigureTestCaseFixture : public framework::Fixture
{
    using T = float;

public:
    void setup()
    {
        _max_batches = 8;
        _cur_batches = 6;
        _target      = compute_target();
        _reference   = compute_reference();
    };

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(0.5f, 1.f);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target()
    {
        AllocatorType allocator{};
        auto          lifetime_mgr = std::make_shared<BlobLifetimeManager>();
        auto          pool_mgr     = std::make_shared<PoolManager>();
        auto          mm           = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

        // Create tensors
        TensorType w1  = create_tensor<TensorType>(TensorShape(128U, 128U), DataType::F32, 1);
        TensorType b1  = create_tensor<TensorType>(TensorShape(128U), DataType::F32, 1);
        TensorType w2  = create_tensor<TensorType>(TensorShape(128U, 24U), DataType::F32, 1);
        TensorType b2  = create_tensor<TensorType>(TensorShape(24U), DataType::F32, 1);
        TensorType src = create_tensor<TensorType>(TensorShape(128U, _max_batches), DataType::F32, 1);
        TensorType fc1 = create_tensor<TensorType>(TensorShape(128U, _max_batches), DataType::F32, 1);
        TensorType dst = create_tensor<TensorType>(TensorShape(24U, _max_batches), DataType::F32, 1);

        // Create and configure function
        FullyConnectedFunction fc_layer_1(mm);
        FullyConnectedFunction fc_layer_2(mm);
        fc_layer_1.configure(&src, &w1, &b1, &fc1);
        fc_layer_2.configure(&fc1, &w2, &b2, &dst);

        // Allocate persistent tensors
        w1.allocator()->allocate();
        b1.allocator()->allocate();
        w2.allocator()->allocate();
        b2.allocator()->allocate();

        // Allocate tensors (1st iteration)
        src.allocator()->allocate();
        fc1.allocator()->allocate();
        dst.allocator()->allocate();

        // Finalize memory manager
        mm->populate(_allocator, 1 /* num_pools */);
        ARM_COMPUTE_ASSERT(mm->lifetime_manager()->are_all_finalized());
        ARM_COMPUTE_ASSERT(mm->pool_manager()->num_pools() == 1);

        // Fill tensors (1st iteration)
        fill(AccessorType(src), 0);
        fill(AccessorType(w1), 1);
        fill(AccessorType(b1), 2);
        fill(AccessorType(w2), 3);
        fill(AccessorType(b2), 4);

        // Compute functions (1st iteration)
        fc_layer_1.run();
        fc_layer_2.run();

        // Update tensor shapes (2nd iteration)
        auto src_padding     = src.allocator()->info().padding();
        auto fc1_padding     = fc1.allocator()->info().padding();
        auto dst_padding     = dst.allocator()->info().padding();
        int  diff            = _max_batches - _cur_batches;
        auto new_src_padding = PaddingSize(src_padding.top, src_padding.right, src_padding.bottom + diff, src_padding.left);
        auto new_fc1_padding = PaddingSize(fc1_padding.top, fc1_padding.right, fc1_padding.bottom + diff, fc1_padding.left);
        auto new_dst_padding = PaddingSize(dst_padding.top, dst_padding.right, dst_padding.bottom + diff, dst_padding.left);
        src.allocator()->info().set_tensor_shape(TensorShape(128U, _cur_batches)).set_is_resizable(true).extend_padding(new_src_padding);
        src.allocator()->info().set_is_resizable(false);
        fc1.allocator()->info().set_tensor_shape(TensorShape(128U, _cur_batches)).set_is_resizable(true).extend_padding(new_fc1_padding);
        fc1.allocator()->info().set_is_resizable(false);
        dst.allocator()->info().set_tensor_shape(TensorShape(24U, _cur_batches)).set_is_resizable(true).extend_padding(new_dst_padding);
        dst.allocator()->info().set_is_resizable(false);

        // Configure FC info
        FullyConnectedLayerInfo fc_info;
        fc_info.retain_internal_weights = true;

        // Configure functions (2nd iteration)
        fc_layer_1.configure(&src, &w1, &b1, &fc1, fc_info);
        fc_layer_2.configure(&fc1, &w2, &b2, &dst, fc_info);

        // Fill tensors (2nd iteration)
        fill(AccessorType(src), 5);

        // Compute functions (2nd iteration)
        fc_layer_1.run();
        fc_layer_2.run();

        return dst;
    }

    SimpleTensor<T> compute_reference()
    {
        // Create reference
        SimpleTensor<T> w1{ TensorShape(128U, 128U), DataType::F32 };
        SimpleTensor<T> b1{ TensorShape(128U), DataType::F32 };
        SimpleTensor<T> w2{ TensorShape(128U, 24U), DataType::F32 };
        SimpleTensor<T> b2{ TensorShape(24U), DataType::F32 };
        SimpleTensor<T> src{ TensorShape(128U, _cur_batches), DataType::F32 };

        // Fill reference
        fill(src, 5);
        fill(w1, 1);
        fill(b1, 2);
        fill(w2, 3);
        fill(b2, 4);

        auto fc1 = reference::fully_connected_layer(src, w1, b1, TensorShape(128U, _cur_batches));
        return reference::fully_connected_layer(fc1, w2, b2, TensorShape(24U, _cur_batches));
    }

protected:
    TensorType      _target{};
    SimpleTensor<T> _reference{};
    AllocatorType   _allocator{};
    unsigned int    _max_batches{};
    unsigned int    _cur_batches{};
};

/** Test case to run a fully connected layer followed by a softmax layer using a blob affinity memory manager,
 *  reconfigure with different shapes and rerun
 *
 * Runs a fully connected convolution layer followed by a softmax layer then reconfigures with different batch size and reruns
 * Shapes of the reconfigure step are smaller that the initial configured step
 */
template <typename TensorType, typename AccessorType, typename AllocatorType, typename FullyConnectedFunction, typename SoftmaxFunction>
class BlobMemoryManagerReconfigure2TestCaseFixture : public framework::Fixture
{
    using T = float;

public:
    void setup()
    {
        _max_batches = 30;
        _cur_batches = 3;
        _target      = compute_target();
        _reference   = compute_reference();
    };

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(0.5f, 1.f);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target()
    {
        AllocatorType allocator{};
        auto          lifetime_mgr = std::make_shared<BlobLifetimeManager>();
        auto          pool_mgr     = std::make_shared<PoolManager>();
        auto          mm           = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

        // Create tensors
        TensorType w   = create_tensor<TensorType>(TensorShape(112U, 8U), DataType::F32, 1);
        TensorType b   = create_tensor<TensorType>(TensorShape(8U), DataType::F32, 1);
        TensorType src = create_tensor<TensorType>(TensorShape(1U, 1U, 112U, _max_batches), DataType::F32, 1);
        TensorType fc  = create_tensor<TensorType>(TensorShape(8U, _max_batches), DataType::F32, 1);
        TensorType dst = create_tensor<TensorType>(TensorShape(8U, _max_batches), DataType::F32, 1);

        // Create and configure function
        FullyConnectedFunction fc_layer(mm);
        SoftmaxFunction        smx_layer(mm);
        fc_layer.configure(&src, &w, &b, &fc);
        smx_layer.configure(&fc, &dst);

        // Allocate persistent tensors
        w.allocator()->allocate();
        b.allocator()->allocate();

        // Allocate tensors (1st iteration)
        src.allocator()->allocate();
        fc.allocator()->allocate();
        dst.allocator()->allocate();

        // Finalize memory manager
        mm->populate(_allocator, 1 /* num_pools */);
        ARM_COMPUTE_ASSERT(mm->lifetime_manager()->are_all_finalized());
        ARM_COMPUTE_ASSERT(mm->pool_manager()->num_pools() == 1);

        // Fill tensors (1st iteration)
        fill(AccessorType(src), 0);
        fill(AccessorType(w), 1);
        fill(AccessorType(b), 2);

        // Compute functions (1st iteration)
        fc_layer.run();
        smx_layer.run();

        // Get padding requirements
        auto fc_padding = fc.allocator()->info().padding();

        // Configure FC info
        FullyConnectedLayerInfo fc_info;
        fc_info.retain_internal_weights = true;

        // Run rest iterations
        for(int i = _max_batches; i >= static_cast<int>(_cur_batches); --i)
        {
            int  diff           = _max_batches - i;
            auto new_fc_padding = PaddingSize(fc_padding.top, fc_padding.right, fc_padding.bottom + diff, fc_padding.left);
            src.allocator()->info().set_tensor_shape(TensorShape(1U, 1U, 112U, i));
            fc.allocator()->info().set_tensor_shape(TensorShape(8U, i)).set_is_resizable(true).extend_padding(new_fc_padding);
            fc.allocator()->info().set_is_resizable(false);
            dst.allocator()->info().set_tensor_shape(TensorShape(8U, i));

            // Configure functions
            fc_layer.configure(&src, &w, &b, &fc, fc_info);
            smx_layer.configure(&fc, &dst);

            // Fill tensors
            fill(AccessorType(src), 3);

            // Compute functions
            fc_layer.run();
            smx_layer.run();
        }

        return dst;
    }

    SimpleTensor<T> compute_reference()
    {
        // Create reference
        SimpleTensor<T> w{ TensorShape(112U, 8U), DataType::F32 };
        SimpleTensor<T> b{ TensorShape(8U), DataType::F32 };
        SimpleTensor<T> src{ TensorShape(1U, 1U, 112U, _cur_batches), DataType::F32 };

        // Fill reference
        fill(src, 3);
        fill(w, 1);
        fill(b, 2);

        auto fc = reference::fully_connected_layer(src, w, b, TensorShape(8U, _cur_batches));
        return reference::softmax_layer(fc, 1.f);
    }

protected:
    TensorType      _target{};
    SimpleTensor<T> _reference{};
    AllocatorType   _allocator{};
    unsigned int    _max_batches{};
    unsigned int    _cur_batches{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNIT_MEMORY_MANAGER */
