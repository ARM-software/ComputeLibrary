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
#include "arm_compute/graph.h"
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;

/** Example demonstrating how to implement YOLOv3 network using the Compute Library's graph API */
class GraphYOLOv3Example : public Example
{
public:
    GraphYOLOv3Example()
        : cmd_parser(), common_opts(cmd_parser), common_params(), graph(0, "YOLOv3")
    {
    }

    bool do_setup(int argc, char **argv) override
    {
        // Parse arguments
        cmd_parser.parse(argc, argv);

        // Consume common parameters
        common_params = consume_common_graph_parameters(common_opts);

        // Return when help menu is requested
        if(common_params.help)
        {
            cmd_parser.print_help(argv[0]);
            return false;
        }

        // Checks
        ARM_COMPUTE_EXIT_ON_MSG(arm_compute::is_data_type_quantized_asymmetric(common_params.data_type), "QASYMM8 not supported for this graph");

        // Print parameter values
        std::cout << common_params << std::endl;

        // Get trainable parameters data path
        std::string data_path = common_params.data_path;

        // Create a preprocessor object
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<TFPreproccessor>(0.f);

        // Create input descriptor
        const TensorShape tensor_shape     = permute_shape(TensorShape(608U, 608U, 3U, 1U), DataLayout::NCHW, common_params.data_layout);
        TensorDescriptor  input_descriptor = TensorDescriptor(tensor_shape, common_params.data_type).set_layout(common_params.data_layout);

        // Set weights trained layout
        const DataLayout weights_layout = DataLayout::NCHW;

        graph << common_params.target
              << common_params.fast_math_hint
              << InputLayer(input_descriptor, get_input_accessor(common_params, std::move(preprocessor), false));
        std::pair<SubStream, SubStream> intermediate_layers = darknet53(data_path, weights_layout);
        graph << ConvolutionLayer(
                  1U, 1U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_53_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv2d_53")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_53_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_53_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_53_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_53_beta.npy"),
                  0.000001f)
              .set_name("conv2d_53/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_53/LeakyRelu")
              << ConvolutionLayer(
                  3U, 3U, 1024U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_54_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv2d_54")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_54_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_54_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_54_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_54_beta.npy"),
                  0.000001f)
              .set_name("conv2d_54/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_54/LeakyRelu")
              << ConvolutionLayer(
                  1U, 1U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_55_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv2d_55")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_55_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_55_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_55_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_55_beta.npy"),
                  0.000001f)
              .set_name("conv2d_55/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_55/LeakyRelu")
              << ConvolutionLayer(
                  3U, 3U, 1024U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_56_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv2d_56")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_56_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_56_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_56_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_56_beta.npy"),
                  0.000001f)
              .set_name("conv2d_56/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_56/LeakyRelu")
              << ConvolutionLayer(
                  1U, 1U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_57_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv2d_57")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_57_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_57_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_57_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_57_beta.npy"),
                  0.000001f)
              .set_name("conv2d_57/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_57/LeakyRelu");
        SubStream route_1(graph);
        graph << ConvolutionLayer(
                  3U, 3U, 1024U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_58_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv2d_58")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_58_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_58_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_58_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_58_beta.npy"),
                  0.000001f)
              .set_name("conv2d_58/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_58/LeakyRelu")
              << ConvolutionLayer(
                  1U, 1U, 255U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_59_w.npy", weights_layout),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_59_b.npy", weights_layout),
                  PadStrideInfo(1, 1, 0, 0))
              .set_name("conv2d_59")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 1.f)).set_name("conv2d_59/Linear")
              << YOLOLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.1f), 80).set_name("Yolo1")
              << OutputLayer(get_output_accessor(common_params, 5));
        route_1 << ConvolutionLayer(
                    1U, 1U, 256U,
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_60_w.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(1, 1, 0, 0))
                .set_name("conv2d_60")
                << BatchNormalizationLayer(
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_59_mean.npy"),
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_59_var.npy"),
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_59_gamma.npy"),
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_59_beta.npy"),
                    0.000001f)
                .set_name("conv2d_59/BatchNorm")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_60/LeakyRelu")
                << UpsampleLayer(Size2D(2, 2), InterpolationPolicy::NEAREST_NEIGHBOR).set_name("Upsample_60");
        SubStream concat_1(route_1);
        concat_1 << ConcatLayer(std::move(route_1), std::move(intermediate_layers.second)).set_name("Route1")
                 << ConvolutionLayer(
                     1U, 1U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_61_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_61")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_60_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_60_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_60_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_60_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_60/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_61/LeakyRelu")
                 << ConvolutionLayer(
                     3U, 3U, 512U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_62_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 1, 1))
                 .set_name("conv2d_62")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_61_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_61_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_61_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_61_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_61/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_62/LeakyRelu")
                 << ConvolutionLayer(
                     1U, 1U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_63_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_63")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_62_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_62_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_62_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_62_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_62/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_63/LeakyRelu")
                 << ConvolutionLayer(
                     3U, 3U, 512U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_64_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 1, 1))
                 .set_name("conv2d_64")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_63_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_63_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_63_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_63_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_63/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_64/LeakyRelu")
                 << ConvolutionLayer(
                     1U, 1U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_65_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_65")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_64_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_64_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_64_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_64_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_65/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_65/LeakyRelu");
        SubStream route_2(concat_1);
        concat_1 << ConvolutionLayer(
                     3U, 3U, 512U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_66_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 1, 1))
                 .set_name("conv2d_66")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_65_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_65_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_65_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_65_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_65/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_66/LeakyRelu")
                 << ConvolutionLayer(
                     1U, 1U, 255U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_67_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_67_b.npy", weights_layout),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_67")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 1.f)).set_name("conv2d_67/Linear")
                 << YOLOLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.1f), 80).set_name("Yolo2")
                 << OutputLayer(get_output_accessor(common_params, 5));
        route_2 << ConvolutionLayer(
                    1U, 1U, 128U,
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_68_w.npy", weights_layout),
                    std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                    PadStrideInfo(1, 1, 0, 0))
                .set_name("conv2d_68")
                << BatchNormalizationLayer(
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_66_mean.npy"),
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_66_var.npy"),
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_66_gamma.npy"),
                    get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_66_beta.npy"),
                    0.000001f)
                .set_name("conv2d_66/BatchNorm")
                << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_68/LeakyRelu")
                << UpsampleLayer(Size2D(2, 2), InterpolationPolicy::NEAREST_NEIGHBOR).set_name("Upsample_68");
        SubStream concat_2(route_2);
        concat_2 << ConcatLayer(std::move(route_2), std::move(intermediate_layers.first)).set_name("Route2")
                 << ConvolutionLayer(
                     1U, 1U, 128U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_69_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_69")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_67_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_67_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_67_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_67_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_67/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_69/LeakyRelu")
                 << ConvolutionLayer(
                     3U, 3U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_70_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 1, 1))
                 .set_name("conv2d_70")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_68_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_68_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_68_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_68_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_68/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_70/LeakyRelu")
                 << ConvolutionLayer(
                     1U, 1U, 128U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_71_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_71")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_69_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_69_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_69_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_69_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_69/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_71/LeakyRelu")
                 << ConvolutionLayer(
                     3U, 3U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_72_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 1, 1))
                 .set_name("conv2d_72")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_70_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_70_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_70_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_70_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_70/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_72/LeakyRelu")
                 << ConvolutionLayer(
                     1U, 1U, 128U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_73_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_73")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_71_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_71_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_71_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_71_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_71/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_73/LeakyRelu")
                 << ConvolutionLayer(
                     3U, 3U, 256U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_74_w.npy", weights_layout),
                     std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                     PadStrideInfo(1, 1, 1, 1))
                 .set_name("conv2d_74")
                 << BatchNormalizationLayer(
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_72_mean.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_72_var.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_72_gamma.npy"),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_72_beta.npy"),
                     0.000001f)
                 .set_name("conv2d_72/BatchNorm")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_74/LeakyRelu")
                 << ConvolutionLayer(
                     1U, 1U, 255U,
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_75_w.npy", weights_layout),
                     get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_75_b.npy", weights_layout),
                     PadStrideInfo(1, 1, 0, 0))
                 .set_name("conv2d_75")
                 << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, 1.f)).set_name("conv2d_75/Linear")
                 << YOLOLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC, 0.1f), 80).set_name("Yolo3")
                 << OutputLayer(get_output_accessor(common_params, 5));

        // Finalize graph
        GraphConfig config;
        config.num_threads = common_params.threads;
        config.use_tuner   = common_params.enable_tuner;
        config.tuner_mode  = common_params.tuner_mode;
        config.tuner_file  = common_params.tuner_file;

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

    std::pair<SubStream, SubStream> darknet53(const std::string &data_path, DataLayout weights_layout)
    {
        graph << ConvolutionLayer(
                  3U, 3U, 32U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_1_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(1, 1, 1, 1))
              .set_name("conv2d_1/Conv2D")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_1_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_1_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_1_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_1_beta.npy"),
                  0.000001f)
              .set_name("conv2d_1/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_1/LeakyRelu")
              << ConvolutionLayer(
                  3U, 3U, 64U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_2_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 1, 1))
              .set_name("conv2d_2/Conv2D")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_2_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_2_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_2_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_2_beta.npy"),
                  0.000001f)
              .set_name("conv2d_2/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_2/LeakyRelu");
        darknet53_block(data_path, "3", weights_layout, 32U);
        graph << ConvolutionLayer(
                  3U, 3U, 128U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_5_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 1, 1))
              .set_name("conv2d_5/Conv2D")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_5_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_5_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_5_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_5_beta.npy"),
                  0.000001f)
              .set_name("conv2d_5/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_5/LeakyRelu");
        darknet53_block(data_path, "6", weights_layout, 64U);
        darknet53_block(data_path, "8", weights_layout, 64U);
        graph << ConvolutionLayer(
                  3U, 3U, 256U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_10_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 1, 1))
              .set_name("conv2d_10/Conv2D")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_10_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_10_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_10_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_10_beta.npy"),
                  0.000001f)
              .set_name("conv2d_10/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_10/LeakyRelu");
        darknet53_block(data_path, "11", weights_layout, 128U);
        darknet53_block(data_path, "13", weights_layout, 128U);
        darknet53_block(data_path, "15", weights_layout, 128U);
        darknet53_block(data_path, "17", weights_layout, 128U);
        darknet53_block(data_path, "19", weights_layout, 128U);
        darknet53_block(data_path, "21", weights_layout, 128U);
        darknet53_block(data_path, "23", weights_layout, 128U);
        darknet53_block(data_path, "25", weights_layout, 128U);
        SubStream layer_36(graph);
        graph << ConvolutionLayer(
                  3U, 3U, 512U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_27_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 1, 1))
              .set_name("conv2d_27/Conv2D")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_27_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_27_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_27_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_27_beta.npy"),
                  0.000001f)
              .set_name("conv2d_27/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_27/LeakyRelu");
        darknet53_block(data_path, "28", weights_layout, 256U);
        darknet53_block(data_path, "30", weights_layout, 256U);
        darknet53_block(data_path, "32", weights_layout, 256U);
        darknet53_block(data_path, "34", weights_layout, 256U);
        darknet53_block(data_path, "36", weights_layout, 256U);
        darknet53_block(data_path, "38", weights_layout, 256U);
        darknet53_block(data_path, "40", weights_layout, 256U);
        darknet53_block(data_path, "42", weights_layout, 256U);
        SubStream layer_61(graph);
        graph << ConvolutionLayer(
                  3U, 3U, 1024U,
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/conv2d_44_w.npy", weights_layout),
                  std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                  PadStrideInfo(2, 2, 1, 1))
              .set_name("conv2d_44/Conv2D")
              << BatchNormalizationLayer(
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_44_mean.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_44_var.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_44_gamma.npy"),
                  get_weights_accessor(data_path, "/cnn_data/yolov3_model/batch_normalization_44_beta.npy"),
                  0.000001f)
              .set_name("conv2d_44/BatchNorm")
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_44/LeakyRelu");
        darknet53_block(data_path, "45", weights_layout, 512U);
        darknet53_block(data_path, "47", weights_layout, 512U);
        darknet53_block(data_path, "49", weights_layout, 512U);
        darknet53_block(data_path, "51", weights_layout, 512U);

        return std::pair<SubStream, SubStream>(layer_36, layer_61);
    }

    void darknet53_block(const std::string &data_path, std::string &&param_path, DataLayout weights_layout,
                         unsigned int filter_size)
    {
        std::string total_path  = "/cnn_data/yolov3_model/";
        std::string param_path2 = arm_compute::support::cpp11::to_string(arm_compute::support::cpp11::stoi(param_path) + 1);
        SubStream   i_a(graph);
        SubStream   i_b(graph);
        i_a << ConvolutionLayer(
                1U, 1U, filter_size,
                get_weights_accessor(data_path, total_path + "conv2d_" + param_path + "_w.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 0, 0))
            .set_name("conv2d_" + param_path + "/Conv2D")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path + "_mean.npy"),
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path + "_var.npy"),
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path + "_gamma.npy"),
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path + "_beta.npy"),
                0.000001f)
            .set_name("conv2d_" + param_path + "/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_" + param_path + "/LeakyRelu")
            << ConvolutionLayer(
                3U, 3U, filter_size * 2,
                get_weights_accessor(data_path, total_path + "conv2d_" + param_path2 + "_w.npy", weights_layout),
                std::unique_ptr<arm_compute::graph::ITensorAccessor>(nullptr),
                PadStrideInfo(1, 1, 1, 1))
            .set_name("conv2d_" + param_path2 + "/Conv2D")
            << BatchNormalizationLayer(
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path2 + "_mean.npy"),
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path2 + "_var.npy"),
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path2 + "_gamma.npy"),
                get_weights_accessor(data_path, total_path + "batch_normalization_" + param_path2 + "_beta.npy"),
                0.000001f)
            .set_name("conv2d_" + param_path2 + "/BatchNorm")
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LEAKY_RELU, 0.1f)).set_name("conv2d_" + param_path2 + "/LeakyRelu");

        graph << EltwiseLayer(std::move(i_a), std::move(i_b), EltwiseOperation::Add).set_name("").set_name("add_" + param_path + "_" + param_path2);
    }
};

/** Main program for YOLOv3
 *
 * Model is based on:
 *      https://arxiv.org/abs/1804.02767
 *      "YOLOv3: An Incremental Improvement"
 *      Joseph Redmon, Ali Farhadi
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
    return arm_compute::utils::run_example<GraphYOLOv3Example>(argc, argv);
}
