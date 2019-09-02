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
#include "CommonGraphOptions.h"

#include "arm_compute/graph/TypeLoader.h"
#include "arm_compute/graph/TypePrinter.h"

#include "support/ToolchainSupport.h"

#include <map>

using namespace arm_compute::graph;

namespace
{
std::pair<unsigned int, unsigned int> parse_validation_range(const std::string &validation_range)
{
    std::pair<unsigned int /* start */, unsigned int /* end */> range = { 0, std::numeric_limits<unsigned int>::max() };
    if(!validation_range.empty())
    {
        std::string       str;
        std::stringstream stream(validation_range);

        // Get first value
        std::getline(stream, str, ',');
        if(stream.fail())
        {
            return range;
        }
        else
        {
            range.first = arm_compute::support::cpp11::stoi(str);
        }

        // Get second value
        std::getline(stream, str);
        if(stream.fail())
        {
            range.second = range.first;
            return range;
        }
        else
        {
            range.second = arm_compute::support::cpp11::stoi(str);
        }
    }
    return range;
}
} // namespace

namespace arm_compute
{
namespace utils
{
::std::ostream &operator<<(::std::ostream &os, const CommonGraphParams &common_params)
{
    std::string false_str = std::string("false");
    std::string true_str  = std::string("true");

    os << "Threads : " << common_params.threads << std::endl;
    os << "Target : " << common_params.target << std::endl;
    os << "Data type : " << common_params.data_type << std::endl;
    os << "Data layout : " << common_params.data_layout << std::endl;
    os << "Tuner enabled? : " << (common_params.enable_tuner ? true_str : false_str) << std::endl;
    os << "Cache enabled? : " << (common_params.enable_cl_cache ? true_str : false_str) << std::endl;
    os << "Tuner mode : " << common_params.tuner_mode << std::endl;
    os << "Tuner file : " << common_params.tuner_file << std::endl;
    os << "Fast math enabled? : " << (common_params.fast_math_hint == FastMathHint::Enabled ? true_str : false_str) << std::endl;
    if(!common_params.data_path.empty())
    {
        os << "Data path : " << common_params.data_path << std::endl;
    }
    if(!common_params.image.empty())
    {
        os << "Image file : " << common_params.image << std::endl;
    }
    if(!common_params.labels.empty())
    {
        os << "Labels file : " << common_params.labels << std::endl;
    }
    if(!common_params.validation_file.empty())
    {
        os << "Validation range : " << common_params.validation_range_start << "-" << common_params.validation_range_end << std::endl;
        os << "Validation file : " << common_params.validation_file << std::endl;
        if(!common_params.validation_path.empty())
        {
            os << "Validation path : " << common_params.validation_path << std::endl;
        }
    }

    return os;
}

CommonGraphOptions::CommonGraphOptions(CommandLineParser &parser)
    : help(parser.add_option<ToggleOption>("help")),
      threads(parser.add_option<SimpleOption<int>>("threads", 1)),
      target(),
      data_type(),
      data_layout(),
      enable_tuner(parser.add_option<ToggleOption>("enable-tuner")),
      enable_cl_cache(parser.add_option<ToggleOption>("enable-cl-cache")),
      tuner_mode(),
      fast_math_hint(parser.add_option<ToggleOption>("fast-math")),
      data_path(parser.add_option<SimpleOption<std::string>>("data")),
      image(parser.add_option<SimpleOption<std::string>>("image")),
      labels(parser.add_option<SimpleOption<std::string>>("labels")),
      validation_file(parser.add_option<SimpleOption<std::string>>("validation-file")),
      validation_path(parser.add_option<SimpleOption<std::string>>("validation-path")),
      validation_range(parser.add_option<SimpleOption<std::string>>("validation-range")),
      tuner_file(parser.add_option<SimpleOption<std::string>>("tuner-file"))
{
    std::set<arm_compute::graph::Target> supported_targets
    {
        Target::NEON,
        Target::CL,
        Target::GC,
    };

    std::set<arm_compute::DataType> supported_data_types
    {
        DataType::F16,
        DataType::F32,
        DataType::QASYMM8,
    };

    std::set<DataLayout> supported_data_layouts
    {
        DataLayout::NHWC,
        DataLayout::NCHW,
    };

    const std::set<CLTunerMode> supported_tuner_modes
    {
        CLTunerMode::EXHAUSTIVE,
        CLTunerMode::NORMAL,
        CLTunerMode::RAPID
    };

    target      = parser.add_option<EnumOption<Target>>("target", supported_targets, Target::NEON);
    data_type   = parser.add_option<EnumOption<DataType>>("type", supported_data_types, DataType::F32);
    data_layout = parser.add_option<EnumOption<DataLayout>>("layout", supported_data_layouts);
    tuner_mode  = parser.add_option<EnumOption<CLTunerMode>>("tuner-mode", supported_tuner_modes, CLTunerMode::NORMAL);

    help->set_help("Show this help message");
    threads->set_help("Number of threads to use");
    target->set_help("Target to execute on");
    data_type->set_help("Data type to use");
    data_layout->set_help("Data layout to use");
    enable_tuner->set_help("Enable OpenCL dynamic tuner");
    enable_cl_cache->set_help("Enable OpenCL program caches");
    tuner_mode->set_help("Configures the time taken by the tuner to tune. Slow tuner produces the most performant LWS configuration");
    fast_math_hint->set_help("Enable fast math");
    data_path->set_help("Path where graph parameters reside");
    image->set_help("Input image for the graph");
    labels->set_help("File containing the output labels");
    validation_file->set_help("File used to validate the graph");
    validation_path->set_help("Path to the validation data");
    validation_range->set_help("Range of the images to validate for (Format : start,end)");
    tuner_file->set_help("File to load/save CLTuner values");
}

CommonGraphParams consume_common_graph_parameters(CommonGraphOptions &options)
{
    FastMathHint fast_math_hint_value = options.fast_math_hint->value() ? FastMathHint::Enabled : FastMathHint::Disabled;
    auto         validation_range     = parse_validation_range(options.validation_range->value());

    CommonGraphParams common_params;
    common_params.help      = options.help->is_set() ? options.help->value() : false;
    common_params.threads   = options.threads->value();
    common_params.target    = options.target->value();
    common_params.data_type = options.data_type->value();
    if(options.data_layout->is_set())
    {
        common_params.data_layout = options.data_layout->value();
    }
    common_params.enable_tuner           = options.enable_tuner->is_set() ? options.enable_tuner->value() : false;
    common_params.enable_cl_cache        = common_params.target == arm_compute::graph::Target::CL ? (options.enable_cl_cache->is_set() ? options.enable_cl_cache->value() : true) : false;
    common_params.tuner_mode             = options.tuner_mode->value();
    common_params.fast_math_hint         = options.fast_math_hint->is_set() ? fast_math_hint_value : FastMathHint::Disabled;
    common_params.data_path              = options.data_path->value();
    common_params.image                  = options.image->value();
    common_params.labels                 = options.labels->value();
    common_params.validation_file        = options.validation_file->value();
    common_params.validation_path        = options.validation_path->value();
    common_params.validation_range_start = validation_range.first;
    common_params.validation_range_end   = validation_range.second;
    common_params.tuner_file             = options.tuner_file->value();

    return common_params;
}
} // namespace utils
} // namespace arm_compute
