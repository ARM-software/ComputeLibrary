/*
 * Copyright (c) 2020-2024 Arm Limited.
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
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/StringUtils.h"
#include "arm_compute/runtime/NEON/functions/NEMaxUnpoolingLayer.h"
#include "arm_compute/runtime/NEON/functions/NEPoolingLayer.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "src/cpu/kernels/CpuMaxUnpoolingLayerKernel.h"
#include "tests/NEON/Accessor.h"
#include "tests/datasets/ShapeDatasets.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
#include "tests/validation/Validation.h"
#include "tests/validation/fixtures/MaxUnpoolingLayerFixture.h"
namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(PoolingLayer)

template <typename T>
using NEMaxUnpoolingLayerFixture = MaxUnpoolingLayerValidationFixture<Tensor, Accessor, NEPoolingLayer, NEMaxUnpoolingLayer, T>;

const auto PoolingLayerIndicesDatasetFPSmall = combine(combine(framework::dataset::make("PoolType", { PoolingType::MAX }), framework::dataset::make("PoolingSize", { Size2D(2, 2) })),
                                                       framework::dataset::make("PadStride", { PadStrideInfo(2, 2, 0, 0), PadStrideInfo(2, 1, 0, 0) }));

TEST_SUITE(Float)
TEST_SUITE(FP32)
FIXTURE_DATA_TEST_CASE(MaxUnpooling, NEMaxUnpoolingLayerFixture<float>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallNoneUnitShapes(), combine(PoolingLayerIndicesDatasetFPSmall,
                                                                                                                   framework::dataset::make("DataType", DataType::F32))),
                                                                                                                   framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })

                                                                                                                  ))
{
    // Validate output
    validate(Accessor(_target), _reference);
}
TEST_SUITE_END() // FP32
#ifdef ARM_COMPUTE_ENABLE_FP16
TEST_SUITE(FP16)
FIXTURE_DATA_TEST_CASE(MaxUnpooling, NEMaxUnpoolingLayerFixture<half>, framework::DatasetMode::PRECOMMIT, combine(combine(datasets::SmallNoneUnitShapes(), combine(PoolingLayerIndicesDatasetFPSmall,
                                                                                                                  framework::dataset::make("DataType", DataType::F16))),
                                                                                                                  framework::dataset::make("DataLayout", { DataLayout::NCHW, DataLayout::NHWC })

                                                                                                                 ))
{
    if(CPUInfo::get().has_fp16())
    {
        // Validate output
        validate(Accessor(_target), _reference);
    }
    else
    {
        ARM_COMPUTE_TEST_INFO("Device does not support fp16 vector operations. Test SKIPPED.");
        framework::ARM_COMPUTE_PRINT_INFO();
    }
}
TEST_SUITE_END() // FP16
#endif           /* ARM_COMPUTE_ENABLE_FP16 */

TEST_SUITE_END() // Float

TEST_SUITE(KernelSelection)

DATA_TEST_CASE(KernelSelection, framework::DatasetMode::ALL,
               combine(framework::dataset::make("CpuExt", std::string("NEON")),
                       framework::dataset::make("DataType", { DataType::F32,
                                                              DataType::F16,
                                                              DataType::QASYMM8,
                                                              DataType::QASYMM8_SIGNED
                                                            })),
               cpu_ext, data_type)
{
    using namespace cpu::kernels;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.sve  = (cpu_ext == "SVE");
    cpu_isa.fp16 = (data_type == DataType::F16);

    const auto *selected_impl = CpuMaxUnpoolingLayerKernel::get_implementation(DataTypeISASelectorData{ data_type, cpu_isa }, cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl);

    std::string expected = lower_string(cpu_ext) + "_" + cpu_impl_dt(data_type) + "_maxunpooling";
    std::string actual   = selected_impl->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}
TEST_SUITE_END() // KernelSelection
TEST_SUITE_END() // PoolingLayer
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
