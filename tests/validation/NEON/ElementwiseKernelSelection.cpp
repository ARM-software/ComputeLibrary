/*
 * Copyright (c) 2022 Arm Limited.
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
#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/cpu/kernels/CpuElementwiseKernel.h"
#include "src/cpu/kernels/CpuElementwiseUnaryKernel.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/framework/datasets/Datasets.h"
namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(NEON)
TEST_SUITE(KernelSelection)

DATA_TEST_CASE(KernelSelection_elementwise_unary, framework::DatasetMode::ALL, concat(
                   combine(framework::dataset::make("CpuExt", std::string("NEON")),
                           framework::dataset::make("DataType", { DataType::F32,
                                                                  DataType::F16,
                                                                  DataType::S32
                                                                })),
                   combine(framework::dataset::make("CpuExt", std::string("SVE")),
                           framework::dataset::make("DataType", { DataType::F32,
                                                                  DataType::F16,
                                                                  DataType::S32
                                                                }))),
               cpu_ext, data_type)
{
    using namespace cpu::kernels;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.sve  = (cpu_ext == "SVE");
    cpu_isa.fp16 = (data_type == DataType::F16);

    const auto *selected_impl = CpuElementwiseUnaryKernel::get_implementation(DataTypeISASelectorData{ data_type, cpu_isa }, cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl);

    std::string expected = lower_string(cpu_ext) + "_" + cpu_impl_dt(data_type) + "_elementwise_unary";
    std::string actual   = selected_impl->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(KernelSelection_elementwise_arithmetic, framework::DatasetMode::ALL, concat(concat(
                                                                                               combine(framework::dataset::make("CpuExt", std::string("NEON")),
                                                                                                       framework::dataset::make("DataType", { DataType::F32,
                                                                                                               DataType::F16,
                                                                                                               DataType::S32,
                                                                                                               DataType::S16,
                                                                                                               DataType::QASYMM8,
                                                                                                               DataType::QASYMM8_SIGNED
                                                                                                                                            })),
                                                                                               combine(framework::dataset::make("CpuExt", std::string("SVE")),
                                                                                                       framework::dataset::make("DataType", { DataType::F32,
                                                                                                               DataType::F16,
                                                                                                               DataType::S32,
                                                                                                               DataType::S16
                                                                                                                                            }))),
                                                                                           combine(framework::dataset::make("CpuExt", std::string("SVE2")),
                                                                                                   framework::dataset::make("DataType", { DataType::QASYMM8,
                                                                                                           DataType::QASYMM8_SIGNED
                                                                                                                                        }))),
               cpu_ext, data_type)
{
    using namespace cpu::kernels;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.sve  = (cpu_ext == "SVE");
    cpu_isa.sve2 = (cpu_ext == "SVE2");
    cpu_isa.fp16 = (data_type == DataType::F16);

    const auto *selected_impl = CpuArithmeticKernel::get_implementation(
                                    ElementwiseDataTypeISASelectorData{ data_type, cpu_isa, static_cast<int>(ArithmeticOperation::ADD) },
                                    cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl);

    std::string expected = lower_string(cpu_ext) + "_" + cpu_impl_dt(data_type) + "_arithmetic";
    std::string actual   = selected_impl->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}

DATA_TEST_CASE(KernelSelection_elementwise_comparison, framework::DatasetMode::ALL, concat(concat(
                                                                                               combine(framework::dataset::make("CpuExt", std::string("NEON")),
                                                                                                       framework::dataset::make("DataType", { DataType::F32,
                                                                                                               DataType::F16,
                                                                                                               DataType::S32,
                                                                                                               DataType::S16,
                                                                                                               DataType::U8,
                                                                                                               DataType::QASYMM8,
                                                                                                               DataType::QASYMM8_SIGNED
                                                                                                                                            })),
                                                                                               combine(framework::dataset::make("CpuExt", std::string("SVE")),
                                                                                                       framework::dataset::make("DataType", { DataType::F32,
                                                                                                               DataType::F16,
                                                                                                               DataType::S32,
                                                                                                               DataType::S16,
                                                                                                               DataType::U8
                                                                                                                                            }))),
                                                                                           combine(framework::dataset::make("CpuExt", std::string("SVE2")),
                                                                                                   framework::dataset::make("DataType", { DataType::QASYMM8,
                                                                                                           DataType::QASYMM8_SIGNED
                                                                                                                                        }))),
               cpu_ext, data_type)
{
    using namespace cpu::kernels;

    cpuinfo::CpuIsaInfo cpu_isa{};
    cpu_isa.neon = (cpu_ext == "NEON");
    cpu_isa.sve  = (cpu_ext == "SVE");
    cpu_isa.sve2 = (cpu_ext == "SVE2");
    cpu_isa.fp16 = (data_type == DataType::F16);

    const auto *selected_impl = CpuComparisonKernel::get_implementation(
                                    ElementwiseDataTypeISASelectorData{ data_type, cpu_isa, static_cast<int>(ComparisonOperation::Equal) },
                                    cpu::KernelSelectionType::Preferred);

    ARM_COMPUTE_ERROR_ON_NULLPTR(selected_impl);

    std::string expected = lower_string(cpu_ext) + "_" + cpu_impl_dt(data_type) + "_comparison";
    std::string actual   = selected_impl->name;

    ARM_COMPUTE_EXPECT_EQUAL(expected, actual, framework::LogLevel::ERRORS);
}

TEST_SUITE_END()
TEST_SUITE_END() // Neon
} // namespace validation
} // namespace test
} // namespace arm_compute
