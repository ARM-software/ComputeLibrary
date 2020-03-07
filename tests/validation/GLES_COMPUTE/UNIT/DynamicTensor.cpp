/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCBufferAllocator.h"
#include "arm_compute/runtime/GLES_COMPUTE/functions/GCNormalizationLayer.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "tests/AssetsLibrary.h"
#include "tests/GLES_COMPUTE/GCAccessor.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/fixtures/UNIT/DynamicTensorFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
#ifndef DOXYGEN_SKIP_THIS
using GCNormLayerWrapper = SimpleFunctionWrapper<MemoryManagerOnDemand, GCNormalizationLayer, IGCTensor>;
template <>
void GCNormLayerWrapper::configure(IGCTensor *src, IGCTensor *dst)
{
    _func.configure(src, dst, NormalizationLayerInfo(NormType::CROSS_MAP, 3));
}
#endif // DOXYGEN_SKIP_THIS

TEST_SUITE(GC)
TEST_SUITE(UNIT)
TEST_SUITE(DynamicTensor)

using BlobMemoryManagementService        = MemoryManagementService<GCBufferAllocator, BlobLifetimeManager, PoolManager, MemoryManagerOnDemand>;
using GCDynamicTensorType3SingleFunction = DynamicTensorType3SingleFunction<GCTensor, GCAccessor, BlobMemoryManagementService, GCNormLayerWrapper>;

/** Tests the memory manager with dynamic input and output tensors.
 *
 *  Create and manage the tensors needed to run a simple function. After the function is executed,
 *  change the input and output size requesting more memory and go through the manage/allocate process.
 *  The memory manager should be able to update the inner structures and allocate the requested memory
 * */
FIXTURE_DATA_TEST_CASE(DynamicTensorType3Single, GCDynamicTensorType3SingleFunction, framework::DatasetMode::ALL,
                       framework::dataset::zip(framework::dataset::make("Level0Shape", { TensorShape(12U, 11U, 3U), TensorShape(256U, 8U, 12U) }),
                                               framework::dataset::make("Level1Shape", { TensorShape(67U, 31U, 15U), TensorShape(11U, 2U, 3U) })))
{
    ARM_COMPUTE_EXPECT(internal_l0.size() == internal_l1.size(), framework::LogLevel::ERRORS);
    ARM_COMPUTE_EXPECT(cross_l0.size() == cross_l1.size(), framework::LogLevel::ERRORS);

    const unsigned int internal_size = internal_l0.size();
    const unsigned int cross_size    = cross_l0.size();
    if(input_l0.total_size() < input_l1.total_size())
    {
        for(unsigned int i = 0; i < internal_size; ++i)
        {
            ARM_COMPUTE_EXPECT(internal_l0[i].size < internal_l1[i].size, framework::LogLevel::ERRORS);
        }
        for(unsigned int i = 0; i < cross_size; ++i)
        {
            ARM_COMPUTE_EXPECT(cross_l0[i].size < cross_l1[i].size, framework::LogLevel::ERRORS);
        }
    }
    else
    {
        for(unsigned int i = 0; i < internal_size; ++i)
        {
            ARM_COMPUTE_EXPECT(internal_l0[i].size == internal_l1[i].size, framework::LogLevel::ERRORS);
        }
        for(unsigned int i = 0; i < cross_size; ++i)
        {
            ARM_COMPUTE_EXPECT(cross_l0[i].size == cross_l1[i].size, framework::LogLevel::ERRORS);
        }
    }
}

TEST_SUITE_END() // DynamicTensor
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // GC
} // namespace validation
} // namespace test
} // namespace arm_compute
