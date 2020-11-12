/*
 * Copyright (c) 2016-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLCHANNELEXTRACT_H
#define ARM_COMPUTE_CLCHANNELEXTRACT_H

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/ICLSimpleFunction.h"

namespace arm_compute
{
class CLCompileContext;
class ICLMultiImage;
class ICLTensor;
using ICLImage = ICLTensor;

/** Basic function to run @ref CLChannelExtractKernel to perform channel extraction.
 *
 * @deprecated This function is deprecated and is intended to be removed in 21.05 release
 *
*/
class CLChannelExtract : public ICLSimpleFunction
{
public:
    /** Initialize the function's source, destination
     *
     * @param[in]  input   The input tensor to extract the channel from. Formats supported: RGB888/RGBA8888/YUYV422/UYVY422
     * @param[in]  channel The channel to extract.
     * @param[out] output  The extracted channel. Must be of U8 format.
     */
    void configure(const ICLTensor *input, Channel channel, ICLTensor *output);
    /** Initialize the function's source, destination
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           The input tensor to extract the channel from. Formats supported: RGB888/RGBA8888/YUYV422/UYVY422
     * @param[in]  channel         The channel to extract.
     * @param[out] output          The extracted channel. Must be of U8 format.
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input, Channel channel, ICLTensor *output);
    /** Initialize the function's source, destination
     *
     * @param[in]  input   The multi-planar input image to extract channel from. Formats supported: NV12/NV21/IYUV/YUV444
     * @param[in]  channel The channel to extract.
     * @param[out] output  The extracted 2D channel. Must be of U8 format.
     */
    void configure(const ICLMultiImage *input, Channel channel, ICLImage *output);
    /** Initialize the function's source, destination
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input           The multi-planar input image to extract channel from. Formats supported: NV12/NV21/IYUV/YUV444
     * @param[in]  channel         The channel to extract.
     * @param[out] output          The extracted 2D channel. Must be of U8 format.
     */
    void configure(const CLCompileContext &compile_context, const ICLMultiImage *input, Channel channel, ICLImage *output);
};
}
#endif /*ARM_COMPUTE_CLCHANNELEXTRACT_H*/
