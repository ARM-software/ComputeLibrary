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
#ifndef ARM_COMPUTE_ITRANSFORMWEIGHTS_H
#define ARM_COMPUTE_ITRANSFORMWEIGHTS_H

#include <atomic>
#include <utility>

namespace arm_compute
{
// Forward declarations
class ITensor;

/** Weights tensor transform interface
 *  In order to identify the different reshape functions, each reshape function has
 * to generate a unique id. We use the following conversion using an unsigned 32bit value:
 *
 * Lower two bits store the target:
 * 00 -> Neon
 * 01 -> CL
 * 11 -> Unused
 *
 * Five bits store the id of the reshape function:
 * 00000 -> FullyConnectedLayerReshapeWeights
 * 00001 -> ConvertFullyConnectedWeights
 * 00010 -> ConvolutionLayerReshapeWeights
 * 00011 -> DepthwiseConvolutionLayerReshapeWeights
 * 00100 -> GEMMReshapeLHSMatrixKernel
 * 00101 -> GEMMReshapeRHSMatrixKernel
 *
 * Rest of the bits are used for identifying special cases such as assembly functions and extra
 * arguments in the reshape kernels.
 *
 * */
class ITransformWeights
{
public:
    /** Default Constructor */
    ITransformWeights() = default;
    /** Default Destructor */
    virtual ~ITransformWeights() = default;
    /** Prevent instances of this class to be copy constructed */
    ITransformWeights(const ITransformWeights &) = delete;
    /** Prevent instances of this class to be copied */
    ITransformWeights &operator=(const ITransformWeights &) = delete;
    /** Allow instances of this class to be move constructed */
    ITransformWeights(ITransformWeights &&other)
    {
        *this = std::move(other);
    }
    /** Allow instances of this class to be moved */
    ITransformWeights &operator=(ITransformWeights &&other)
    {
        if(this != &other)
        {
            _num_refcount = other._num_refcount.load();
            _reshape_run  = other._reshape_run;
        }
        return *this;
    }

    /** Get a pointer to the transformed weights
     *
     * @return The pointer to the transformed ITensor weights
     */
    virtual ITensor *get_weights() = 0;
    /** Function that returns a unique id of the reshape function
     *
     * @return The computed unique id
     */
    virtual uint32_t uid() = 0;
    /** Run the transformation function */
    virtual void run() = 0;
    /** Release transformed weights memory */
    virtual void release() = 0;
    /** Increase the object's refcount */
    void increase_refcount()
    {
        ++_num_refcount;
    }

    /** Decrease the object's refcount and return the updated value
     *
     * @return The updated refcount
     * */
    int32_t decrease_refcount()
    {
        return --_num_refcount;
    }

    /** Function that returns a flag on whether the weights are reshaped or not
     *
     * @return True if the function is reshaped
     */
    bool is_reshape_run()
    {
        return _reshape_run;
    }

protected:
    std::atomic<int32_t> _num_refcount{ 0 };
    bool                 _reshape_run{ false };
};
} // arm_compute

#endif /*ARM_COMPUTE_ITRANSFORMWEIGHTS_H */
