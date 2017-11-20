/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_FIXED_POINT_CL_TARGET
#define ARM_COMPUTE_TEST_FIXED_POINT_CL_TARGET

#include "arm_compute/runtime/CL/CLScheduler.h"

#include "tests/Globals.h"
#include "tests/Types.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace
{
template <typename TensorType, typename AccessorType, typename T>
void compute_target_impl(const TensorShape &shape, DataType dt, FixedPointOp op, int fixed_point_position, TensorType &src, TensorType &dst)
{
    std::string fixed_point_operation_kernel;
#ifndef EMBEDDED_KERNELS
    std::cout << "EMBEDDED_KERNELS NOT DEFINED" << std::endl;

    fixed_point_operation_kernel += "#include \"fixed_point.h\"\n";
#endif /* EMBEDDED_KERNELS */
    fixed_point_operation_kernel +=
        "__kernel void fixed_point_operation_qs8(                                                                 \n"
        "   __global char* src,                                                                                   \n"
        "   __global char* dst)                                                                                   \n"
        "{                                                                                                        \n"
        "   char16 in = vload16(0, src + get_global_id(0) * 16);                                                  \n"
        "   if(FIXED_POINT_OP == 0)                                                                               \n"
        "   {                                                                                                     \n"
        "       vstore16(EXP_OP_EXPAND(in, DATA_TYPE, 16, FIXED_POINT_POS), 0, dst + get_global_id(0) * 16);      \n"
        "   }                                                                                                     \n"
        "   else if(FIXED_POINT_OP == 1)                                                                          \n"
        "   {                                                                                                     \n"
        "       vstore16(INVSQRT_OP_EXPAND(in, DATA_TYPE, 16, FIXED_POINT_POS), 0, dst + get_global_id(0) * 16);  \n"
        "   }                                                                                                     \n"
        "   else                                                                                                  \n"
        "   {                                                                                                     \n"
        "       vstore16(LOG_OP_EXPAND(in, DATA_TYPE, 16, FIXED_POINT_POS), 0, dst + get_global_id(0) * 16);      \n"
        "   }                                                                                                     \n"
        "}                                                                                                        \n"
        "\n";

    // Set build options
    std::string build_opts = "-DFIXED_POINT_POS=" + support::cpp11::to_string(fixed_point_position);
    build_opts += " -DDATA_TYPE=qs8";

    // Fill tensors.
    int min = 0;
    int max = 0;
    switch(op)
    {
        case(FixedPointOp::EXP):
            min = -(1 << (fixed_point_position - 1));
            max = (1 << (fixed_point_position - 1));
            build_opts += " -DFIXED_POINT_OP=0";
            break;
        case(FixedPointOp::INV_SQRT):
            min = 1;
            max = (dt == DataType::QS8) ? 0x7F : 0x7FFF;
            build_opts += " -DFIXED_POINT_OP=1";
            break;
        case(FixedPointOp::LOG):
            min = (1 << (fixed_point_position - 1));
            max = (dt == DataType::QS8) ? 0x3F : 0x3FFF;
            build_opts += " -DFIXED_POINT_OP=2";
            break;
        default:
            ARM_COMPUTE_ERROR("Fixed point operation not supported");
            break;
    }

    std::uniform_int_distribution<> distribution(min, max);
    library->fill(AccessorType(src), distribution, 0);

    std::vector<std::string> sources;

#ifndef EMBEDDED_KERNELS
    build_opts += " -I" + CLKernelLibrary::get().get_kernel_path();
#else  /* EMBEDDED_KERNELS */
    sources.push_back(CLKernelLibrary::get().get_program_source("fixed_point.h"));
#endif /* EMBEDDED_KERNELS */

    sources.push_back(fixed_point_operation_kernel);

    // Create program
    ::cl::Program program(sources);

    // Build program
    program.build(build_opts.c_str());

    ::cl::Kernel kernel(program, "fixed_point_operation_qs8", nullptr);

    unsigned int idx = 0;
    kernel.setArg(idx++, src.cl_buffer());
    kernel.setArg(idx++, dst.cl_buffer());

    ::cl::NDRange gws(shape[0] / 16, 1, 1);
    CLScheduler::get().queue().enqueueNDRangeKernel(kernel, 0, gws);
}
} // namespace
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FIXED_POINT_TARGET */
