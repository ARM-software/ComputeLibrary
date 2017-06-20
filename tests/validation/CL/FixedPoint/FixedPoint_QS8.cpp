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
#include "CL/CLAccessor.h"
#include "Globals.h"
#include "TensorLibrary.h"
#include "TypePrinter.h"
#include "Utils.h"
#include "validation/Datasets.h"
#include "validation/Reference.h"
#include "validation/Validation.h"

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"

#include "arm_compute/core/CL/ICLTensor.h"

#include "boost_wrapper.h"

#include <random>
#include <string>

using namespace arm_compute;
using namespace arm_compute::test;
using namespace arm_compute::test::cl;
using namespace arm_compute::test::validation;

namespace
{
const float tolerance_exp     = 1.0f; /**< Tolerance value for comparing reference's output against implementation's output  (exponential)*/
const float tolerance_invsqrt = 4.0f; /**< Tolerance value for comparing reference's output against implementation's output (inverse square-root) */
const float tolerance_log     = 5.0f; /**< Tolerance value for comparing reference's output against implementation's output (logarithm) */

/** Compute Neon fixed point operation for signed 8bit fixed point.
 *
 * @param[in] shape Shape of the input and output tensors.
 *
 * @return Computed output tensor.
 */
CLTensor compute_fixed_point_op(const TensorShape &shape, int fixed_point_position, FixedPointOp op)
{
    std::string fixed_point_operation_kernel;
#ifndef EMBEDDED_KERNELS
    fixed_point_operation_kernel += "#include \"fixed_point.h\"\n";
#endif
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

    // Create tensors
    CLTensor src = create_tensor<CLTensor>(shape, DataType::QS8, 1, fixed_point_position);
    CLTensor dst = create_tensor<CLTensor>(shape, DataType::QS8, 1, fixed_point_position);

    // Allocate tensors
    src.allocator()->allocate();
    dst.allocator()->allocate();

    BOOST_TEST(!src.info()->is_resizable());
    BOOST_TEST(!dst.info()->is_resizable());

    // Set build options
    std::string build_opts = "-DFIXED_POINT_POS=" + val_to_string<int>(fixed_point_position);
    build_opts += " -DDATA_TYPE=qs8";

    // Fill tensors.
    int min = 0;
    int max = 0;
    switch(op)
    {
        case FixedPointOp::EXP:
            min = -(1 << (fixed_point_position - 1));
            max = (1 << (fixed_point_position - 1));
            build_opts += " -DFIXED_POINT_OP=0";
            break;
        case FixedPointOp::INV_SQRT:
            min = 1;
            max = 0x7F;
            build_opts += " -DFIXED_POINT_OP=1";
            break;
        case FixedPointOp::LOG:
            min = (1 << (fixed_point_position - 1));
            max = 0x3F;
            build_opts += " -DFIXED_POINT_OP=2";
            break;
        default:
            ARM_COMPUTE_ERROR("Operation not supported");
    }

    std::uniform_int_distribution<> distribution(min, max);
    library->fill(CLAccessor(src), distribution, 0);

    std::vector<std::string> sources;

#ifndef EMBEDDED_KERNELS
    build_opts += " -I" + CLKernelLibrary::get().get_kernel_path();
#else
    sources.push_back(CLKernelLibrary::get().get_program_source("fixed_point.h"));
#endif /* EMBEDDED_KERNELS */

    sources.push_back(fixed_point_operation_kernel);

    // Create program
    ::cl::Program program = ::cl::Program(sources);

    // Build program
    program.build(build_opts.c_str());

    ::cl::Kernel kernel = ::cl::Kernel(program, "fixed_point_operation_qs8", nullptr);

    unsigned int idx = 0;
    kernel.setArg(idx++, src.cl_buffer());
    kernel.setArg(idx++, dst.cl_buffer());

    ::cl::NDRange gws(shape[0] / 16, 1, 1);
    CLScheduler::get().queue().enqueueNDRangeKernel(kernel, 0, gws);

    return dst;
}
} // namespace

#ifndef DOXYGEN_SKIP_THIS
BOOST_AUTO_TEST_SUITE(CL)
BOOST_AUTO_TEST_SUITE(FixedPoint)
BOOST_AUTO_TEST_SUITE(QS8)

BOOST_AUTO_TEST_SUITE(Exp)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunSmall, Small1DShape() * boost::unit_test::data::xrange(1, 6), shape, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_fixed_point_op(shape, fixed_point_position, FixedPointOp::EXP);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fixed_point_operation(shape, DataType::QS8, DataType::QS8, FixedPointOp::EXP, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_exp);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Log)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunSmall, Small1DShape() * boost::unit_test::data::xrange(3, 6), shape, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_fixed_point_op(shape, fixed_point_position, FixedPointOp::LOG);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fixed_point_operation(shape, DataType::QS8, DataType::QS8, FixedPointOp::LOG, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_log);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Invsqrt)

BOOST_TEST_DECORATOR(*boost::unit_test::label("precommit") * boost::unit_test::label("nightly"))
BOOST_DATA_TEST_CASE(RunSmall, Small1DShape() * boost::unit_test::data::xrange(1, 6), shape, fixed_point_position)
{
    // Compute function
    CLTensor dst = compute_fixed_point_op(shape, fixed_point_position, FixedPointOp::INV_SQRT);

    // Compute reference
    RawTensor ref_dst = Reference::compute_reference_fixed_point_operation(shape, DataType::QS8, DataType::QS8, FixedPointOp::INV_SQRT, fixed_point_position);

    // Validate output
    validate(CLAccessor(dst), ref_dst, tolerance_invsqrt);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
#endif
