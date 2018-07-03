/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_EXAMPLES_UTILS_COMMON_GRAPH_OPTIONS
#define ARM_COMPUTE_EXAMPLES_UTILS_COMMON_GRAPH_OPTIONS

#include "utils/command_line/CommandLineOptions.h"
#include "utils/command_line/CommandLineParser.h"

#include "arm_compute/graph/TypeLoader.h"
#include "arm_compute/graph/TypePrinter.h"

namespace arm_compute
{
namespace utils
{
/** Structure holding all the common graph parameters */
struct CommonGraphParams
{
    bool                             help{ false };
    int                              threads{ 0 };
    arm_compute::graph::Target       target{ arm_compute::graph::Target::NEON };
    arm_compute::DataType            data_type{ DataType::F32 };
    arm_compute::DataLayout          data_layout{ DataLayout::NCHW };
    bool                             enable_tuner{ false };
    arm_compute::graph::FastMathHint fast_math_hint{ arm_compute::graph::FastMathHint::DISABLED };
    std::string                      data_path{};
    std::string                      image{};
    std::string                      labels{};
    std::string                      validation_file{};
    std::string                      validation_path{};
    unsigned int                     validation_range_start{ 0 };
    unsigned int                     validation_range_end{ std::numeric_limits<unsigned int>::max() };
};

/** Formatted output of the CommonGraphParams type
 *
 * @param[out] os            Output stream.
 * @param[in]  common_params Common parameters to output
 *
 * @return Modified output stream.
 */
::std::ostream &operator<<(::std::ostream &os, const CommonGraphParams &common_params);

/** Common command line options used to configure the graph examples
 *
 * The options in this object get populated when "parse()" is called on the parser used to construct it.
 * The expected workflow is:
 *
 * CommandLineParser parser;
 * CommonOptions options( parser );
 * parser.parse(argc, argv);
 */
class CommonGraphOptions
{
public:
    /** Constructor
     *
     * @param[in,out] parser A parser on which "parse()" hasn't been called yet.
     */
    CommonGraphOptions(CommandLineParser &parser);
    /** Prevent instances of this class from being copy constructed */
    CommonGraphOptions(const CommonGraphOptions &) = delete;
    /** Prevent instances of this class from being copied */
    CommonGraphOptions &operator=(const CommonGraphOptions &) = delete;

    ToggleOption                           *help;             /**< Show help option */
    SimpleOption<int>                      *threads;          /**< Number of threads option */
    EnumOption<arm_compute::graph::Target> *target;           /**< Graph execution target */
    EnumOption<arm_compute::DataType>      *data_type;        /**< Graph data type */
    EnumOption<arm_compute::DataLayout>    *data_layout;      /**< Graph data layout */
    ToggleOption                           *enable_tuner;     /**< Enable tuner */
    ToggleOption                           *fast_math_hint;   /**< Fast math hint */
    SimpleOption<std::string>              *data_path;        /**< Trainable parameters path */
    SimpleOption<std::string>              *image;            /**< Image */
    SimpleOption<std::string>              *labels;           /**< Labels */
    SimpleOption<std::string>              *validation_file;  /**< Validation file */
    SimpleOption<std::string>              *validation_path;  /**< Validation data path */
    SimpleOption<std::string>              *validation_range; /**< Validation range */
};

/** Consumes the common graph options and creates a structure containing any information
 *
 * @param[in] options Options to consume
 *
 * @return Structure containing the commnon graph parameters
 */
CommonGraphParams consume_common_graph_parameters(CommonGraphOptions &options);
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_EXAMPLES_UTILS_COMMON_GRAPH_OPTIONS */
