/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/graph.h"
#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/Utils.h"
#endif /* ARM_COMPUTE_CL */
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

class GraphVanillaTransformerExample : public Example
{
public: 
    GraphVanillaTransformerExample(): cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "Vanilla_Transformer")
    {
    }
    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);
        cmd_parser.validate();

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if (common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;


        /*
            Args:
                d_model: the number of expected features in the input (required).
                nhead: the number of heads in the multiheadattention models (required).
                dim_feedforward: the dimension of the feedforward network model (default=2048).
                dropout: the dropout value (default=0.1).
                activation: the activation function of the intermediate layer, can be a string
                    ("relu" or "gelu") or a unary callable. Default: relu
                layer_norm_eps: the eps value in layer normalization components (default=1e-5).
                batch_first: If ``True``, then the input and output tensors are provided
                    as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
                norm_first: if ``True``, layer norm is done prior to attention and feedforward
                    operations, respectively. Otherwise it's done after. Default: ``False`` (after).
                bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
                    bias. Default: ``True``.
        */
       // Model parameters
       /*constexpr unsigned int d_model   = 512U;     // Dim layer output 
       constexpr unsigned int h         = 8U;       // Parallel attention (Heads)
       constexpr unsigned int d_ff      = 2024U;    // Dim feedforward
       constexpr unsigned int d_q       = 64U;      // Dim query, 512U/8U
       constexpr unsigned int d_k       = 64U;      // Dim key, 512U/8U
       constexpr unsigned int d_v       = 64U;      // Dim value, 512U/8U
       constexpr float        P_drop    = 0.1f;     // Dropout rate

       
       constexpr unsigned int seq_src   = 25000U;   // Input token sequence
       constexpr unsigned int seq_tgt   = 25000U;   // Output token sequence.
       constexpr unsigned int bs        = 1U;       // Batch size*/

       // Compute library best operate on NHWC(default) layout
       //const auto operation_layout = common_params.data_layout;

       // Create input tensor
       const TensorShape src_tensor = TensorShape(5U);

       // Maybe permute input data layout to target operation layout  



        TensorDescriptor input_descriptor = TensorDescriptor(src_tensor, common_params.data_type);

        // Set graph hints
        graph << common_params.target << common_params.fast_math_hint;

        // Encode Input
        graph << InputLayer(input_descriptor, get_input_accessor(common_params)).set_name("in1")
              << OutputLayer(get_output_accessor(common_params)).set_name("out1") 
              << InputLayer(input_descriptor, get_input_accessor(common_params)).set_name("in2");
            //<< TokenEmbeddingLayer(TokenEmbeddingLayerInfo(d_model),get_weights_accessor(data_path,"data/npy/token_embedding.npy"));

        // Decode Input
        // Finalize graph
        GraphConfig config;

        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;
        config.mlgo_file   = common_params.mlgo_file;

        graph.finalize(common_params.target, config);

        return true;
    }

    void do_run() override
    {
        // Run graph
        graph.run();
    }

private:
    CommandLineParser  cmd_parser;
    CommonGraphOptions common_opts;
    CommonGraphParams  common_params;
    Stream             graph;
};

/** Main program for Vanilla Transformer
 *
 * Model is based on:
 *      "Attention Is All You Need". 
 *      Ashish Vaswani, Noam Shazeer,Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser,Illia Polosukhin. 2017.
 *
 * @note To list all the possible arguments execute the binary appended with the --help option
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    return arm_compute::utils::run_example<GraphVanillaTransformerExample>(argc, argv);
}
