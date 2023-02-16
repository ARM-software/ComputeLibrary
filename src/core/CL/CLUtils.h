/*
 * Copyright (c) 2020-2021, 2023 Arm Limited.
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

#ifndef ARM_COMPUTE_CL_CLUTILS_H
#define ARM_COMPUTE_CL_CLUTILS_H

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/experimental/IPostOp.h"

namespace arm_compute
{
class TensorShape;
class CLBuildOptions;
class ITensorInfo;

/** OpenCL Image2D types */
enum class CLImage2DType
{
    ReadOnly,
    WriteOnly
};

/** Create a cl::Image2D object from an OpenCL buffer
 *
 * @note The following conditions are required to create a OpenCL image object from OpenCL buffer,
 *       -# The platform should support the OpenCL cl_khr_image2d_from_buffer extension
 *       -# The stride Y for the input1 should satisfy the OpenCL pitch alignment requirement
 *       -# input width should be less or equal to (CL_DEVICE_IMAGE2D_MAX_WIDTH * 4)
 *       -# input height should be less or equal to CL_DEVICE_IMAGE2D_MAX_HEIGHT
 *
 * It is user responsibility to ensure the above conditions are satisfied since no checks are performed within this function
 *
 * @param[in] ctx             cl::Context object
 * @param[in] buffer          cl::Buffer object from which the OpenCL image2d object is created
 * @param[in] shape2d         2D tensor shape
 * @param[in] data_type       DataType to use. Only supported: F32,F16
 * @param[in] image_row_pitch Image row pitch (a.k.a. stride Y) to be used in the image2d object
 * @param[in] image_type      Image 2D type (@ref CLImage2DType)
 *
 * @return cl::Image2D object
 */
cl::Image2D create_image2d_from_buffer(const cl::Context &ctx, const cl::Buffer &buffer, const TensorShape &shape2d, DataType data_type, size_t image_row_pitch, CLImage2DType image_type);

namespace experimental
{
/** @name (EXPERIMENTAL_POST_OPS)
 * @{
 */

/** Manage validation, building and configurations of PostOp CL kernels */
class PostOpCLKernelUtils final
{
public:
    /** CL kernel name postfix for post ops */
    using NamePostfix = std::string;
    /** CL kernels that supports post ops assign each post op to a 'slot', in accordance with the postfix
     * For example, for a kernel with postfix '_act_prelu_eltwiseadd', there are 3 slots
     * slot 1: (unary) activation, slot 2: pRelu, slot 3: elementwise addition
     *
     * Some kernels may allow some slots to be optional, to support multiple combinations of post op sequences.
     * In such cases, we need to explicitly set up a mapping between each post op and the slots for that kernel.
     * For example, suppose we have 2 kernels with postfixes: _eltwiseadd_prelu, _act_eltwiseadd_act_prelu, where the activations in the
     * second kernel are optional. Say we want to support an eltwise addition, followed by a prelu (sequence { eltwiseadd, prelu }).
     * Now we can choose which one of the 2 kernels to use, since they both support this post op sequence.
     * We can either:
     *  1. assign the elementwise to slot 1 and prelu to slot 2 of kernel 1
     *  { { Eltwise_Add, PRelu } -> {"_eltwise_act", {1, 2} } } or
     *  2. assign the elementwise to slot 2 and prelu to slot 4 of kernel 1
     *  { { Eltwise_Add, PRelu } -> {"_act_eltwiseadd_act_prelu", {2, 4} } }
     */
    using Slots  = std::vector<unsigned int>;
    using Config = std::map<PostOpTypeSequence, std::tuple<NamePostfix, Slots>>;

public:
    explicit PostOpCLKernelUtils(const Config &config);

    /** Check if post op argument tensor shapes are compliant
     * All post ops must not alter the shape of the original dst tensor (even after broadcasting)
     *
     * @param[in] dst      Dst tensor to apply the post ops to
     * @param[in] post_ops Post ops
     *
     * @return true if shapes are compliant and false otherwise
     */
    static bool are_post_op_shapes_compliant(const ITensorInfo *dst, const experimental::PostOpList<ITensorInfo *> &post_ops);
    /** Check if the post op sequence is supported in the current configuration
     *
     * @param[in] post_ops Post ops
     *
     * @return true if the post op sequence is supported and false otherwise
     */
    bool is_post_op_sequence_supported(const PostOpList<ITensorInfo *> &post_ops) const;
    /** Helper function to set PostOp related build options
     * @note Convention
     *      1. Each post op "slot" is prefixed with "P<slot number>", followed by the usual parameters for that post op.
     *      E.g. If the first slot is an activation, we need to pass 3 definitions in this way:
     *          -P1_ACTIVATION_TYPE=...  -P1_ACTIVATION_A_VAL=...   -P1_ACTIVATION_B_VAL=...
     *
     *      2. For multi-ary post ops, to pass the position of the previous op's dest tensor,
     *         we append "_X_POS_<pos>" to the post op type.
     *      E.g. for a single post op add(dst, x), where dst is the result of the main op.
     *         In this case, the position of the previous op's dest is 0, so we pass
     *         -P1_ELTWISE_OP=ADD_X_POS_0
     *
     * @param[out] built_opts OpenCL kernel build options
     * @param[in]  post_ops   Post ops
     *
     */
    void set_post_ops_cl_build_options(CLBuildOptions &built_opts, const PostOpList<ITensorInfo *> &post_ops) const;
    /** Helper function to set PostOp kernel name
     *
     * @param[out] kernel_name OpenCL kernel name
     * @param[in]  post_ops    Post ops
     *
     */
    void set_post_ops_cl_kernel_name(std::string &kernel_name, const PostOpList<ITensorInfo *> &post_ops) const;

private:
    Config _supported_config{};
};
/** @} */ // end of group (EXPERIMENTAL_POST_OPS)

} // namespace experimental

} // arm_compute

#endif /* ARM_COMPUTE_CL_CLUTILS_H */
