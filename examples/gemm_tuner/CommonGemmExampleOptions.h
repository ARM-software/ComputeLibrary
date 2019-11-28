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
#ifndef ARM_COMPUTE_EXAMPLES_GEMM_TUNER_COMMON_GEMM_EXAMPLE_OPTIONS
#define ARM_COMPUTE_EXAMPLES_GEMM_TUNER_COMMON_GEMM_EXAMPLE_OPTIONS

#include "utils/command_line/CommandLineOptions.h"
#include "utils/command_line/CommandLineParser.h"

namespace gemm_tuner
{
/** Structure holding all the common gemm example parameters */
struct CommonGemmExampleParams
{
    size_t M{ 100 }; /**< Number of lhs matrix rows */
    size_t N{ 100 }; /**< Number of rhs matrix columns */
    size_t K{ 50 };  /**< Number of lhs matrix columns/rhs matrix rows */
    size_t B{ 1 };   /**< Batch size */
};

/** Formatted output of the CommonGemmExampleParams type
 *
 * @param[out] os            Output stream.
 * @param[in]  common_params Common parameters to output
 *
 * @return Modified output stream.
 */
::std::ostream &operator<<(::std::ostream &os, const CommonGemmExampleParams &common_params);

/** Common command line options used to configure the gemm examples
 *
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 */
class CommonGemmExampleOptions
{
public:
    /** Constructor
     *
     * @param[in,out] parser A parser on which "parse()" hasn't been called yet.
     */
    CommonGemmExampleOptions(arm_compute::utils::CommandLineParser &parser);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CommonGemmExampleOptions(const CommonGemmExampleOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CommonGemmExampleOptions &operator=(const CommonGemmExampleOptions &) = delete;
    /** Allow instances of this class to be moved */
    CommonGemmExampleOptions(CommonGemmExampleOptions &&) = default;
    /** Allow instances of this class to be moved */
    CommonGemmExampleOptions &operator=(CommonGemmExampleOptions &&) = default;
    /** Default destructor */
    ~CommonGemmExampleOptions() = default;

    arm_compute::utils::ToggleOption         *help; /**< Show help option */
    arm_compute::utils::SimpleOption<size_t> *M;    /**< Number of lhs matrix rows option */
    arm_compute::utils::SimpleOption<size_t> *N;    /**< Number of rhs matrix columns option */
    arm_compute::utils::SimpleOption<size_t> *K;    /**< Number of lhs matrix columns/rhs matrix rows option */
    arm_compute::utils::SimpleOption<size_t> *B;    /**< Batch size option */
};

/** Consumes the common gemm example options and creates a structure containing all information
 *
 * @param[in] options Options to consume
 *
 * @return Structure containing the common gemm example parameters
 */
CommonGemmExampleParams consume_common_gemm_example_parameters(const CommonGemmExampleOptions &options);
} // namespace gemm_tuner
#endif /* ARM_COMPUTE_EXAMPLES_GEMM_TUNER_COMMON_GEMM_EXAMPLE_OPTIONS */
