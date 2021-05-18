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
#include "CommonGemmExampleOptions.h"

namespace gemm_tuner
{
using namespace arm_compute;
using namespace utils;

::std::ostream &operator<<(::std::ostream &os, const CommonGemmExampleParams &common_params)
{
    os << "M : " << common_params.M << std::endl;
    os << "N : " << common_params.N << std::endl;
    os << "K : " << common_params.K << std::endl;
    os << "B : " << common_params.B << std::endl;
    os << "Data type : " << common_params.data_type << std::endl;
    os << "OpenCL tuner mode : " << common_params.tuner_mode << std::endl;
    return os;
}

CommonGemmExampleOptions::CommonGemmExampleOptions(arm_compute::utils::CommandLineParser &parser, arm_compute::DataType default_data_type)
    : help(parser.add_option<ToggleOption>("help")),
      M(parser.add_positional_option<SimpleOption<size_t>>("M", 100)),
      N(parser.add_positional_option<SimpleOption<size_t>>("N", 100)),
      K(parser.add_positional_option<SimpleOption<size_t>>("K", 50)),
      B(parser.add_positional_option<SimpleOption<size_t>>("B", 1)),
      data_type(),
      tuner_mode()
{
    const std::set<DataType> supported_data_types
    {
        DataType::F16,
        DataType::F32,
        DataType::QASYMM8,
    };

    const std::set<CLTunerMode> supported_tuner_modes
    {
        CLTunerMode::EXHAUSTIVE,
        CLTunerMode::NORMAL,
        CLTunerMode::RAPID
    };

    ARM_COMPUTE_ERROR_ON_MSG(supported_data_types.find(default_data_type) == supported_data_types.end(), "Default data type unsupported");

    data_type  = parser.add_option<EnumOption<DataType>>("type", supported_data_types, default_data_type);
    tuner_mode = parser.add_option<EnumOption<CLTunerMode>>("tuner-mode", supported_tuner_modes, CLTunerMode::RAPID);

    help->set_help("Show this help message.");
    M->set_help("Number of lhs matrix rows.");
    N->set_help("Number of rhs matrix columns.");
    K->set_help("Number of lhs matrix columns/rhs matrix rows.");
    B->set_help("Batch size.");
    data_type->set_help("Data type to use");
    tuner_mode->set_help("OpenCL tuner mode");
}

CommonGemmExampleParams consume_common_gemm_example_parameters(const CommonGemmExampleOptions &options)
{
    CommonGemmExampleParams common_params;
    common_params.M          = options.M->value();
    common_params.N          = options.N->value();
    common_params.K          = options.K->value();
    common_params.B          = options.B->value();
    common_params.data_type  = options.data_type->value();
    common_params.tuner_mode = options.tuner_mode->value();
    return common_params;
}
} // namespace gemm_tuner
