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
    GraphVanillaTransformerExample()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "Vanilla_Transformer")
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
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Model parameters
        constexpr unsigned int d_model    = 768U;   // Dim layer output
        constexpr unsigned int d_vocab    = 30522U; // Vocaboary size
        constexpr unsigned int d_segemnt  = 2U;     // Sentence segmentation size
        constexpr unsigned int d_position = 512U;   // Pretrained positional encoding length
        constexpr unsigned int h          = 12U;    // Parallel attention (Heads)
        constexpr float        eps        = 1e-12;  // Layer normalization eplision
        constexpr unsigned int d_ff       = 3072U;  // Dim feedforward
        /*constexpr unsigned int d_q         = 64U;      // Dim query, 512U/8U
        constexpr unsigned int d_k           = 64U;      // Dim key, 512U/8U
        constexpr unsigned int d_v           = 64U;      // Dim value, 512U/8U
        constexpr float        P_drop        = 0.1f;     // Dropout rate


        constexpr unsigned int seq_src       = 25000U;   // Input token sequence
        constexpr unsigned int seq_tgt       = 25000U;   // Output token sequence.
        constexpr unsigned int bs            = 1U;       // Batch size*/

        // Compute library best operate on NHWC(default) layout
        //const auto operation_layout = common_params.data_layout;

        // Create input tensor
        const TensorShape src_tensor = TensorShape(7U);

        // Data layout
        const DataLayout operation_layout = DataLayout::NCHW;

        TensorDescriptor input_descriptor = TensorDescriptor(src_tensor, common_params.data_type);

        // Set graph hints
        graph << common_params.target << common_params.fast_math_hint;

        // Text preprocessor
        //std::unique_ptr<IPreprocessor> WP_preproccessor     = std::make_unique<WordPiecePreprocessor>(common_params.vocabulary);
        std::unique_ptr<IPreprocessor> at2_preproccessor = std::make_unique<atoiPreprocessor>();

        // Encode Input
        graph << InputLayer(input_descriptor, get_token_accessor(common_params),
                            get_segment_accessor(common_params.segment, move(at2_preproccessor)))
                     .set_name("in1")

              << EmbeddingLayer(EmbeddingLayerInfo(d_model,
                                                   d_vocab,
                                                   d_segemnt,
                                                   d_position,
                                                   true /*Use pretrained positional encoding*/,
                                                   ConvertPolicy::SATURATE),
                                get_weights_accessor(data_path, "/token_embedding.npy", operation_layout),
                                get_weights_accessor(data_path, "/segment_embedding.npy", operation_layout),
                                get_weights_accessor(data_path, "/positional_embedding.npy", operation_layout))
                     .set_name("tkemb1");

        add_encoder_block(data_path, d_model, h, eps, d_ff);

        graph << OutputLayer(get_output_accessor(common_params)).set_name("out1");

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

    void add_encoder_block(std::string  data_path,
                           unsigned int d_model, unsigned int h, float eps, unsigned int d_ff)
    {
        SubStream without_attention(graph);
        SubStream with_attention(graph);

        with_attention
            /* Self Attention */
            << MultiHeadLinearLayer(LinearLayerInfo(d_model), get_weights_accessor(data_path, "/layer_0/query_weight.npy"),
                                    get_weights_accessor(data_path, "/layer_0/query_bias.npy"),
                                    get_weights_accessor(data_path, "/layer_0/key_weight.npy"),
                                    get_weights_accessor(data_path, "/layer_0/key_bias.npy"),
                                    get_weights_accessor(data_path, "/layer_0/value_weight.npy"),
                                    get_weights_accessor(data_path, "/layer_0/value_bias.npy"))
            << MultiHeadAttentionLayer(MultiHeadAttentionLayerInfo(d_model, h)).set_name("mha1");

        graph << EltwiseLayer(std::move(with_attention), std::move(without_attention), EltwiseOperation::Add).set_name("add_4_norm_attention");

        /* Self output */
        graph << LayerNormLayer(LayerNormLayerInfo(0 /*Window::DimX*/, eps));

        SubStream without_ff(graph);
        SubStream with_ff(graph);
        /* Self Intermediate(Feed Forward)*/
        with_ff << LinearLayer(LinearLayerInfo(d_ff, TensorShape(d_model, d_ff) /*weight*/,
                                               TensorShape(d_ff) /*bias*/),
                               get_weights_accessor(data_path, "/layer_0/ff_weight_0.npy"),
                               get_weights_accessor(data_path, "/layer_0/ff_bias_0.npy"))
                << ActivationLayer(ActivationLayerInfo(ActivationFunction::GELU))
                << LinearLayer(LinearLayerInfo(d_model, TensorShape(d_ff, d_model) /*weight*/,
                                               TensorShape(d_model) /*bias*/),
                               get_weights_accessor(data_path, "/layer_0/ff_weight_1.npy"),
                               get_weights_accessor(data_path, "/layer_0/ff_bias_1.npy"));

        graph << EltwiseLayer(std::move(with_ff), std::move(without_ff), EltwiseOperation::Add).set_name("add_4_norm_ff");

        /* Output*/
        graph << LayerNormLayer(LayerNormLayerInfo(0 /*Window::DimX*/, eps));
    }
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
