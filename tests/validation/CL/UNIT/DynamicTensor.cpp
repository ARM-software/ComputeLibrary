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
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/CL/CLBufferAllocator.h"
#include "arm_compute/runtime/CL/functions/CLConvolutionLayer.h"
#include "arm_compute/runtime/CL/functions/CLL2NormalizeLayer.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "src/core/CL/kernels/CLFillBorderKernel.h"
#include "src/core/CL/kernels/CLIm2ColKernel.h"
#include "src/core/CL/kernels/CLL2NormalizeLayerKernel.h"
#include "src/core/CL/kernels/CLReductionOperationKernel.h"
#include "src/core/CL/kernels/CLWeightsReshapeKernel.h"
#include "tests/AssetsLibrary.h"
#include "tests/CL/CLAccessor.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/UNIT/DynamicTensorFixture.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
constexpr AbsoluteTolerance<float> absolute_tolerance_float(0.0001f); /**< Absolute Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
RelativeTolerance<float>           tolerance_f32(0.1f);               /**< Tolerance value for comparing reference's output against implementation's output for DataType::F32 */
constexpr float                    tolerance_num = 0.07f;             /**< Tolerance number */
} // namespace

#ifndef DOXYGEN_SKIP_THIS
using CLL2NormLayerWrapper = SimpleFunctionWrapper<MemoryManagerOnDemand, CLL2NormalizeLayer, ICLTensor>;
template <>
void CLL2NormLayerWrapper::configure(ICLTensor *src, ICLTensor *dst)
{
    _func.configure(src, dst, 0, 0.0001f);
}
#endif // DOXYGEN_SKIP_THIS
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(DynamicTensor)

using BlobMemoryManagementService        = MemoryManagementService<CLBufferAllocator, BlobLifetimeManager, PoolManager, MemoryManagerOnDemand>;
using CLDynamicTensorType3SingleFunction = DynamicTensorType3SingleFunction<CLTensor, CLAccessor, BlobMemoryManagementService, CLL2NormLayerWrapper>;

/** Tests the memory manager with dynamic input and output tensors.
 *
 *  Create and manage the tensors needed to run a simple function. After the function is executed,
 *  change the input and output size requesting more memory and go through the manage/allocate process.
 *  The memory manager should be able to update the inner structures and allocate the requested memory
 * */
FIXTURE_DATA_TEST_CASE(DynamicTensorType3Single, CLDynamicTensorType3SingleFunction, framework::DatasetMode::ALL,
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

using CLDynamicTensorType3ComplexFunction = DynamicTensorType3ComplexFunction<CLTensor, CLAccessor, BlobMemoryManagementService, CLConvolutionLayer>;
/** Tests the memory manager with dynamic input and output tensors.
 *
 *  Create and manage the tensors needed to run a complex function. After the function is executed,
 *  change the input and output size requesting more memory and go through the manage/allocate process.
 *  The memory manager should be able to update the inner structures and allocate the requested memory
 * */
FIXTURE_DATA_TEST_CASE(DynamicTensorType3Complex, CLDynamicTensorType3ComplexFunction, framework::DatasetMode::ALL,
                       framework::dataset::zip(framework::dataset::zip(framework::dataset::zip(framework::dataset::zip(
                                                                                                   framework::dataset::make("InputShape", { std::vector<TensorShape>{ TensorShape(12U, 12U, 16U), TensorShape(64U, 64U, 16U) } }),
                                                                                                   framework::dataset::make("WeightsManager", { TensorShape(3U, 3U, 16U, 5U) })),
                                                                                               framework::dataset::make("BiasShape", { TensorShape(5U) })),
                                                                       framework::dataset::make("OutputShape", { std::vector<TensorShape>{ TensorShape(12U, 12U, 5U), TensorShape(64U, 64U, 5U) } })),
                                               framework::dataset::make("PadStrideInfo", { PadStrideInfo(1U, 1U, 1U, 1U) })))
{
    for(unsigned int i = 0; i < num_iterations; ++i)
    {
        run_iteration(i);
        validate(CLAccessor(dst_target), dst_ref, tolerance_f32, tolerance_num, absolute_tolerance_float);
    }
}

using CLDynamicTensorType2PipelineFunction = DynamicTensorType2PipelineFunction<CLTensor, CLAccessor, BlobMemoryManagementService, CLConvolutionLayer>;
/** Tests the memory manager with dynamic input and output tensors.
 *
 *  Create and manage the tensors needed to run a pipeline. After the function is executed, resize the input size and rerun.
 */
FIXTURE_DATA_TEST_CASE(DynamicTensorType2Pipeline, CLDynamicTensorType2PipelineFunction, framework::DatasetMode::ALL,
                       framework::dataset::make("InputShape", { std::vector<TensorShape>{ TensorShape(12U, 12U, 6U), TensorShape(128U, 128U, 6U) } }))
{
}

TEST_SUITE_END() // DynamicTensor
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
