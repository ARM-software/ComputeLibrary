/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/CL/CLCompileContext.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Macros.h"
#include "tests/validation/Validation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
TEST_SUITE(CL)
TEST_SUITE(UNIT)
TEST_SUITE(CompileContext)

TEST_CASE(CompileContextCache, framework::DatasetMode::ALL)
{
    // Create compile context
    CLCompileContext compile_context(CLKernelLibrary::get().context(), CLKernelLibrary::get().get_device());

    // Check if the program cache is empty
    ARM_COMPUTE_EXPECT(compile_context.get_built_programs().size() == 0, framework::LogLevel::ERRORS);

    // Create a kernel using the compile context
    const std::string kernel_name  = "floor_layer";
    const std::string program_name = CLKernelLibrary::get().get_program_name(kernel_name);
    std::pair<std::string, bool> kernel_src = CLKernelLibrary::get().get_program(program_name);
    const std::string kernel_path = CLKernelLibrary::get().get_kernel_path();

    std::set<std::string> build_opts;
    build_opts.emplace("-DDATA_TYPE=float");
    build_opts.emplace("-DVEC_SIZE=16");
    compile_context.create_kernel(kernel_name, program_name, kernel_src.first, kernel_path, build_opts, kernel_src.second);

    // Check if the program is stored in the cache
    ARM_COMPUTE_EXPECT(compile_context.get_built_programs().size() == 1, framework::LogLevel::ERRORS);

    // Try to build the same program and check if the program cache stayed the same
    compile_context.create_kernel(kernel_name, program_name, kernel_src.first, kernel_path, build_opts, kernel_src.second);
    ARM_COMPUTE_EXPECT(compile_context.get_built_programs().size() == 1, framework::LogLevel::ERRORS);
}

TEST_SUITE_END() // CompileContext
TEST_SUITE_END() // UNIT
TEST_SUITE_END() // CL
} // namespace validation
} // namespace test
} // namespace arm_compute
