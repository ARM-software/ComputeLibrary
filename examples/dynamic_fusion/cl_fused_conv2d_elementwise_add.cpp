/*
 * Copyright (c) 2022 Arm Limited.
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

/// @example dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp
/// @copybrief example_dynamic_fusion_cl_conv2d_elementwise_add
///
/// @page example_dynamic_fusion_cl_conv2d_elementwise_add Dynamic Fusion Example: Conv2d + Elementwise Addition (OpenCL target)
/// This example demonstrates how to fuse a Conv2d with an Addition using the new OperatorGraph API, and to run it with the Async Composite Operator

#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
#error "This example needs to be built with -DARM_COMPUTE_CL"
#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/experimental/ClWorkload.h"
#include "arm_compute/core/experimental/OperatorGraph.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTuner.h"
#include "arm_compute/runtime/experimental/ClCompositeOperator.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "utils/TypePrinter.h"

#include "utils/Utils.h"

#include <cstdlib>

using namespace arm_compute;
using namespace utils;
using namespace arm_compute::experimental::dynamic_fusion;

#define TICK(clock_name) \
    auto clock_name##_tick = std::chrono::high_resolution_clock::now();
#define TOCK(clock_name, measurement_map)                                               \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>(clock_name##_tock - clock_name##_tick);
#define TOCK_AVG(clock_name, measurement_map, num_iterations)                           \
    auto clock_name##_tock                 = std::chrono::high_resolution_clock::now(); \
    measurement_map["\"" #clock_name "\""] = duration_cast<microseconds>((clock_name##_tock - clock_name##_tick) / (num_iterations));

using std::chrono::duration_cast;
using std::chrono::microseconds;

class ClFusedConv2dEltwiseAddExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        size_t       ih;
        size_t       iw;
        size_t       ifm;
        size_t       wh;
        size_t       ww;
        size_t       ofm;
        size_t       tuner_choice;
        unsigned int pad_x;
        unsigned int pad_y;
        if(argc < 10)
        {
            // Print help
            std::cout << "Usage:  ./cl_fused_conv2d_elementwise_add ih iw ifm wh ww ofm tuner_choice(0=Disable, 1=Rapid, 2=Normal, 3=Exhaustive) pad_x pad_y\n";
            std::cout << "Too few or no input_matrices provided. Using shape config = SRGAN_0, tuner_choice=2\n\n";
            ih           = 512;
            iw           = 512;
            ifm          = 64;
            wh           = 1;
            ww           = 1;
            ofm          = 3;
            tuner_choice = 2;
            pad_x        = 0;
            pad_y        = 0;
        }
        else
        {
            ih           = strtol(argv[1], nullptr, 10);
            iw           = strtol(argv[2], nullptr, 10);
            ifm          = strtol(argv[3], nullptr, 10);
            wh           = strtol(argv[4], nullptr, 10);
            ww           = strtol(argv[5], nullptr, 10);
            ofm          = strtol(argv[6], nullptr, 10);
            tuner_choice = strtol(argv[7], nullptr, 10);
            pad_x        = strtol(argv[8], nullptr, 10);
            pad_y        = strtol(argv[9], nullptr, 10);
        }

        CLTuner *tuner_to_use;
        switch(tuner_choice)
        {
            case 0:
            {
                tuner_to_use = nullptr;
                break;
            }
            case 1:
            {
                tuner.set_tuner_mode(CLTunerMode::RAPID);
                tuner_to_use = &tuner;
                break;
            }
            case 3:
            {
                tuner.set_tuner_mode(CLTunerMode::EXHAUSTIVE);
                tuner_to_use = &tuner;
                break;
            }
            case 2:
            default:
            {
                tuner.set_tuner_mode(CLTunerMode::NORMAL);
                tuner_to_use = &tuner;
                break;
            }
        }
        CLScheduler::get().default_init(tuner_to_use);

        TICK(startup_time);
        /* Computation:
         * out = add_desc(addend, conv2d1x1(direct_conv)(input, weights, bias))
         */
        const auto data_type   = DataType::F32;
        const auto data_layout = DataLayout::NHWC;

        const auto t_input_shape     = TensorShape(ifm, iw, ih);
        const auto t_weight_shape    = TensorShape(ifm, ww, wh, ofm);
        const auto t_bias_shape      = TensorShape(ofm);
        const auto t_l1_addend_shape = TensorShape(ofm, iw);

        std::cout << "input_shape: " << t_input_shape << std::endl;
        std::cout << "weight_shape: " << t_weight_shape << std::endl;
        std::cout << "bias_shape: " << t_bias_shape << std::endl;
        std::cout << "addend_shape: " << t_l1_addend_shape << std::endl;

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// @section describe_workload_using_operator_graph Describe the workload to run using OperatorGraph
        /// OperatorGraph is a graph of Tensors and Operators. Let's first default-construct it
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Construct OperatorGraph
        // [Construct OperatorGraph]
        OperatorGraph op_graph;
        // [Construct OperatorGraph]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// @subsection add_conv2d Add the first operator (root operator) Conv2d
        /// The first operator to be added to the graph is called the "root operator" of the entire graph.
        /// @note As of now, operators need to be inserted according to their dependency order. This is because output tensor auto-initialization occurs during construction time.
        ///       Later this might be changed to allow out-of-order insertion.

        /// Before we insert the operator, we need to initialize the required TensorInfo objects.
        /// We can choose not to initialize an output TensorInfo; if so, they will be auto-initialized during the construction of the OperatorGraph
        /// The "t_acc_info" is the TensorInfo of the accumulator tensor, which is the output tensor of our first operator conv2d
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp  Initialize Conv2d TensorInfo
        // [Initialize Conv2d TensorInfo]
        auto t_input_info  = TensorInfo(t_input_shape, 1, data_type, data_layout);
        auto t_weight_info = TensorInfo(t_weight_shape, 1, data_type, data_layout);
        auto t_bias_info   = TensorInfo(t_bias_shape, 1, data_type, data_layout);
        auto t_acc_info    = TensorInfo();
        // [Initialize Conv2d TensorInfo]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// Next we associate the TensorInfo with the OpTensor s created in the op_graph.
        /// @note The associated TensorInfo objects must be in scope and remain valid until the ClWorkload building is completed

        /// @note The associated TensorInfo objects must be declard as non-const, since they may be updated during the OperatorGraph construction

        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp  Add OpTensors
        // [Add OpTensors]
        const auto op_t_input  = add_tensor(op_graph, t_input_info);
        const auto op_t_weight = add_tensor(op_graph, t_weight_info);
        const auto op_t_bias   = add_tensor(op_graph, t_bias_info);
        const auto op_t_acc    = add_tensor(op_graph, t_acc_info);
        // [Add OpTensors]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// Finally we add the Conv2d operator to op_graph. The Conv2dDescriptor contains all the TOSA-compliant attribute parameters
        /// The add_op... group of functions accept the OpTensors created by the add_tensor function, and return an Operator handle.
        /// This handle can be used to further query and modify the operator inside the OperatorGraph after its creation
        /// For example, here we use the handle to force the ConvolutionMethod to be Direct Convolution
        /// @note The force_conv2d_method is only for debug purpose for now, as the end user is not expected to decide on the ConvolutionMethod

        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp  Add Conv2d Operator
        // [Add Conv2d Operator]
        Conv2dDescriptor conv2d_desc{ Padding2D{ pad_x, pad_x, pad_y, pad_y } };
        auto             conv2d = add_op_conv2d(op_graph, conv2d_desc, op_t_input, op_t_weight, op_t_bias, op_t_acc);
        force_conv2d_method(op_graph, conv2d, ConvolutionMethod::DIRECT); // Only for debug purposes
        // [Add Conv2d Operator]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// @subsection add_elementwise_add Add the second operator Elementwise Add
        /// This is similar to adding the first operator to op_graph, except that we link the two operators together by their common tensor,
        /// namely the accumulator tensor op_t_acc, which is the output of conv2d and the input (lhs) of the addition
        /// @note At the moment, it is recommended to always declare a separate TensorInfo (even if empty) for each OpTensor.
        ///       For example, here op_t_dst could be associated with op_t_acc info as they are the same,
        ///       but we still recommend creating a separate object.

        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp  Add Elementwise Add Operator
        // [Add Elementwise Add Operator]
        auto          t_l1_addend_info = TensorInfo(t_l1_addend_shape, 1, data_type, data_layout);
        auto          t_dst_info       = TensorInfo();
        const auto    op_t_l1_addend   = add_tensor(op_graph, t_l1_addend_info);
        const auto    op_t_dst         = add_tensor(op_graph, t_dst_info);
        AddDescriptor add_desc{};
        add_op_elementwise_add(op_graph, add_desc, op_t_acc, op_t_l1_addend, op_t_dst);
        // [Add Elementwise Add Operator]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// @section build_clworkload Build ClWorkload
        /// ClWorkload is an intermediate object which contains all the built kernel codes and all other descriptors on how to schedule them
        /// We build ClWorkload from the op_graph object that we just described
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp  Build ClWorkload
        // [Build ClWorkload]
        const ClWorkloadContext workload_ctx
        {
            GpuInfo{ CLScheduler::get().target() }
        };
        ClWorkload workload;
        build(workload, op_graph, workload_ctx);
        // [Build ClWorkload]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// @section run_fused_op_with_clcompositeoperator Run the fused operator workload with ClCompositeOperator
        /// @subsection configure_and_validate_clcompositeoperator Validate ClWorkload and Configure ClCompositeOperator
        /// After ClWorkload is built, we need to configure it with the Compute Library runtime ClCompositeOperator to run it.
        /// Optionally we can explicitly validate the workload to check if the workload has been built successfully.
        /// The validate is automatically run inside configure and would throw if it fails.
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Construct ClCompositeOperator
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp  Validate and configure ClCompositeOperator
        // [Validate and configure ClCompositeOperator]
        const auto success = ClCompositeOperator::validate(workload); // Optional
        op.configure(CLKernelLibrary::get().get_compile_context(), workload);
        // [Validate and configure ClCompositeOperator]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// @subsection run_clcompositeoperator Run ClCompositeOperator
        /// Construct the runtime CLTensor s with backing memory
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Construct CLTensor objects

        /// Initialize, allocate and fill the CLTensor objects
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Initialize, Allocate and Fill CLTensor objects
        // [Initialize, Allocate and Fill CLTensor objects]
        t_input.allocator()->init(t_input_info);
        t_weight.allocator()->init(t_weight_info);
        t_bias.allocator()->init(t_bias_info);
        t_l1_addend.allocator()->init(t_dst_info);
        t_dst.allocator()->init(t_dst_info);

        t_input.allocator()->allocate();
        t_weight.allocator()->allocate();
        t_bias.allocator()->allocate();
        t_l1_addend.allocator()->allocate();
        t_dst.allocator()->allocate();

        fill_random_tensor(t_input, -1.f, 1.f);
        fill_random_tensor(t_weight, -1.f, 1.f);
        fill_random_tensor(t_l1_addend, -1.f, 1.f);
        // [Initialize, Allocate and Fill CLTensor objects]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// The OpTensorBinding creates a mapping from the OpTensor handles that we created early to the real CLTensors
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Create OpTensorBinding
        // [Create OpTensorBinding]
        OpTensorBinding op_tensors({ { op_t_input, &t_input },
            { op_t_weight, &t_weight },
            { op_t_bias, &t_bias },
            { op_t_l1_addend, &t_l1_addend },
            { op_t_dst, &t_dst }
        });
        // [Create OpTensorBinding]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// Bind the CLTensor objects to the prepare_pack_map and run_pack_map, which are used to prepare and run the op
        /// This step additionally creates empty auxiliary CLTensor objects if any, and contain them inside a ClAuxTensorData aux_tensor_data
        /// @note This step associates all the CLTensors contained in op_tensors and aux_tensor_data, with prepare_pack_map and run_pack_map
        ///       Make sure these CLTensors remain valid as long as the two pack_maps are still in use

        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Construct ClAuxTensorData
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Construct TensorPackMaps
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Bind Tensors
        // [Bind Tensors]
        bind_tensors(aux_tensor_data, prepare_pack_map, run_pack_map, workload, op_tensors);
        // [Bind Tensors]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// Initialize and Allocate Auxiliary CLTensor objects.
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Initialize and Allocate Auxiliary CLTensor objects
        // [Initialize and Allocate Auxiliary CLTensor objects]
        for(auto tensor_data : aux_tensor_data.get_tensors())
        {
            tensor_data.tensor->allocator()->init(tensor_data.tensor_info);
            tensor_data.tensor->allocator()->allocate();
        }
        // [Initialize and Allocate Auxiliary CLTensor objects]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// Run the ClCompositeOperator prepare job. This performs any jobs that are required for the first run, like
        /// reshaping tensors for a more performant format.
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Prepare ClCompositeOperator
        // [Prepare ClCompositeOperator]
        op.prepare(prepare_pack_map);
        // [Prepare ClCompositeOperator]

        /// @page example_dynamic_fusion_cl_conv2d_elementwise_add
        /// At last, we run our operator
        /// @snippet dynamic_fusion/cl_fused_conv2d_elementwise_add.cpp Run ClCompositeOperator
        // [Run ClCompositeOperator]
        op.run(run_pack_map);
        // [Run ClCompositeOperator]
        TOCK(startup_time, measurements);
        return true;
    }
    void do_run() override
    {
        // Run the fused op
        op.run(run_pack_map);

        // Make sure all the OpenCL jobs are done executing:
        CLScheduler::get().sync();
    }

    void do_teardown() override
    {
        for(auto m : measurements)
        {
            std::cout << m.first << ": " << m.second.count() << "us" << std::endl;
        }
    }

private:
    // [Construct CLTensor objects]
    CLTensor t_input{};
    CLTensor t_weight{};
    CLTensor t_bias{};
    CLTensor t_l1_addend{};
    CLTensor t_dst{};
    // [Construct CLTensor objects]
    // [Construct ClAuxTensorData]
    ClAuxTensorData aux_tensor_data{};
    // [Construct ClAuxTensorData]
    // [Construct TensorPackMaps]
    TensorPackMap prepare_pack_map{};
    TensorPackMap run_pack_map{};
    // [Construct TensorPackMaps]
    // [Construct ClCompositeOperator]
    ClCompositeOperator op{};
    // [Construct ClCompositeOperator]
    CLTuner tuner{};
    std::map<std::string, std::chrono::microseconds> measurements{};
};

/** Main program for sgemm test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<ClFusedConv2dEltwiseAddExample>(argc, argv);
}

#undef TICK
#undef TOCK
#undef TOCK_AVG
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */