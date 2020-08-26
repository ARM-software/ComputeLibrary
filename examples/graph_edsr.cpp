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

#include "arm_compute/graph/Utils.h"

#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/Utils.h"

#include "graph_edsr.h"

using namespace arm_compute::graph;
using namespace arm_compute::utils;

class GraphEdsrExample : public Example
{
public:
    GraphEdsrExample()
        : cmd_parser(), common_opts(cmd_parser), common_params()
    {
        expected_output_filename = cmd_parser.add_option<SimpleOption<std::string>>("expected-output-filename", "");
        expected_output_filename->set_help("Name of npy file containing the expected output to validate the graph output.");
    }

    GraphEdsrExample(const GraphEdsrExample &) = delete;
    GraphEdsrExample &operator=(const GraphEdsrExample &) = delete;
    ~GraphEdsrExample() override                          = default;

    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        ARM_COMPUTE_EXIT_ON_MSG(common_params.data_type != DataType::QASYMM8, "Only QASYMM8 is supported for this graph example");

        // Print parameter values
        std::cout << common_params << std::endl;

        model.setup(common_params, *expected_output_filename);

        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;

        context.set_config(config);

        auto pass_manager = create_default_pass_manager(common_params.target, config);
        manager.finalize_graph(model.graph(), context, pass_manager, common_params.target);

        return true;
    }

    void do_run() override
    {
        manager.execute_graph(model.graph());
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;

    GraphContext context{};
    GraphManager manager{};

    SimpleOption<std::string> *expected_output_filename{ nullptr };

    GraphEdsr model{};
};

/** Internal implementation of UINT8 EDSR with some modifications from the paper.
 * The sub-pixel convolution has been replaced with a deconvolution layer. This
 * operation is mathematically the same.
 *
 * Convolution replaced by deconvolution:
 *      https://arxiv.org/abs/1609.07009
 *      "Is the deconvolution layer the same as a convolutional layer?"
 *      Wenzhe Shi, Jose Caballero, Lucas Theis, Ferenc Huszar, Andrew Aitken, Christian Ledig, Zehan Wang
 *
 * Original model is:
 *      https://arxiv.org/abs/1707.02921
 *      "Enhanced Deep Residual Networks for Single Image Super-Resolution"
 *      Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 *
 * @return Return code
 */
int main(int argc, char **argv)
{
    return run_example<GraphEdsrExample>(argc, argv);
}
