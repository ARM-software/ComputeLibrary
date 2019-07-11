/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/CLTunerTypes.h"

namespace arm_compute
{
namespace utils
{
/* ![Common graph examples parameters] */
/* Common graph parameters
 *
 * --help             : Print the example's help message.
 * --threads          : The number of threads to be used by the example during execution.
 * --target           : Execution target to be used by the examples. Supported target options: NEON, CL, GC.
 * --type             : Data type to be used by the examples. Supported data type options: QASYMM8, F16, F32.
 * --layout           : Data layout to be used by the examples. Supported data layout options : NCHW, NHWC.
 * --enable-tuner     : Toggle option to enable the OpenCL dynamic tuner.
 * --enable-cl-cache  : Toggle option to load the prebuilt opencl kernels from a cache file.
 * --fast-math        : Toggle option to enable the fast math option.
 * --data             : Path that contains the trainable parameter files of graph layers.
 * --image            : Image to load and operate on. Image types supported: PPM, JPEG, NPY.
 * --labels           : File that contains the labels that classify upon.
 * --validation-file  : File that contains a list of image names with their corresponding label id (e.g. image0.jpg 5).
 *                      This is used to run the graph over a number of images and report top-1 and top-5 metrics.
 * --validation-path  : The path where the validation images specified in the validation file reside.
 * --validation-range : The range of the images to validate from the validation file (e.g 0,9).
 *                      If not specified all the images will be validated.
 * --tuner-file       : The file to store the OpenCL dynamic tuner tuned parameters.
 *
 * Note that data, image and labels options should be provided to perform an inference run on an image.
 * Note that validation-file and validation-path should be provided to perform a graph accuracy estimation.
 * Note GLES target is not supported for most of the networks.
 *
 * Example execution commands:
 *
 * Execute a single inference given an image and a file containing the correspondence between label ids and human readable labels:
 * ./graph_vgg16 --data=data/ --target=cl --layout=nhwc --image=kart.jpeg --labels=imagenet1000_clsid_to_human.txt
 *
 * Perform a graph validation on a list of images:
 * ./graph_vgg16 --data=data/ --target=neon --threads=4 --layout=nchw --validation-file=val.txt --validation-path=ilsvrc_test_images/
 *
 * File formats:
 *
 * Validation file should be a plain file containing the names of the images followed by the correct label id.
 * For example:
 *
 * image0.jpeg 882
 * image1.jpeg 34
 * image2.jpeg 354
 *
 * Labels file should be a plain file where each line is the respective human readable label (counting starts from 0).
 * For example:
 *
 * 0: label0_name            label0_name
 * 1: label1_name     or     label1_name
 * 2: label2_name            label2_name
 */
/* ![Common graph examples parameters] */

/** Structure holding all the common graph parameters */
struct CommonGraphParams
{
    bool                             help{ false };
    int                              threads{ 0 };
    arm_compute::graph::Target       target{ arm_compute::graph::Target::NEON };
    arm_compute::DataType            data_type{ DataType::F32 };
    arm_compute::DataLayout          data_layout{ DataLayout::NHWC };
    bool                             enable_tuner{ false };
    bool                             enable_cl_cache{ false };
    arm_compute::CLTunerMode         tuner_mode{ CLTunerMode::NORMAL };
    arm_compute::graph::FastMathHint fast_math_hint{ arm_compute::graph::FastMathHint::Disabled };
    std::string                      data_path{};
    std::string                      image{};
    std::string                      labels{};
    std::string                      validation_file{};
    std::string                      validation_path{};
    std::string                      tuner_file{};
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
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CommonGraphOptions(const CommonGraphOptions &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CommonGraphOptions &operator=(const CommonGraphOptions &) = delete;
    /** Allow instances of this class to be moved */
    CommonGraphOptions(CommonGraphOptions &&) = default;
    /** Allow instances of this class to be moved */
    CommonGraphOptions &operator=(CommonGraphOptions &&) = default;
    /** Default destructor */
    ~CommonGraphOptions() = default;

    ToggleOption                           *help;             /**< Show help option */
    SimpleOption<int>                      *threads;          /**< Number of threads option */
    EnumOption<arm_compute::graph::Target> *target;           /**< Graph execution target */
    EnumOption<arm_compute::DataType>      *data_type;        /**< Graph data type */
    EnumOption<arm_compute::DataLayout>    *data_layout;      /**< Graph data layout */
    ToggleOption                           *enable_tuner;     /**< Enable tuner */
    ToggleOption                           *enable_cl_cache;  /**< Enable opencl kernels cache */
    SimpleOption<arm_compute::CLTunerMode> *tuner_mode;       /**< Tuner mode */
    ToggleOption                           *fast_math_hint;   /**< Fast math hint */
    SimpleOption<std::string>              *data_path;        /**< Trainable parameters path */
    SimpleOption<std::string>              *image;            /**< Image */
    SimpleOption<std::string>              *labels;           /**< Labels */
    SimpleOption<std::string>              *validation_file;  /**< Validation file */
    SimpleOption<std::string>              *validation_path;  /**< Validation data path */
    SimpleOption<std::string>              *validation_range; /**< Validation range */
    SimpleOption<std::string>              *tuner_file;       /**< File to load/store the tuner's values from */
};

/** Consumes the common graph options and creates a structure containing any information
 *
 * @param[in] options Options to consume
 *
 * @return Structure containing the common graph parameters
 */
CommonGraphParams consume_common_graph_parameters(CommonGraphOptions &options);
} // namespace utils
} // namespace arm_compute
#endif /* ARM_COMPUTE_EXAMPLES_UTILS_COMMON_GRAPH_OPTIONS */
