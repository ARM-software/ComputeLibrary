/*
 * Copyright (c) 2019, 2021 Arm Limited.
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

#ifndef ARM_COMPUTE_IWEIGHTSMANAGER_H
#define ARM_COMPUTE_IWEIGHTSMANAGER_H

#include "arm_compute/core/ITensor.h"
#include "arm_compute/runtime/ITransformWeights.h"

#include <map>

namespace arm_compute
{
/** Weights manager interface to handle weights transformations */
class IWeightsManager
{
public:
    /** Constructor */
    IWeightsManager();
    /** Default Destructor */
    virtual ~IWeightsManager() = default;
    /** Prevent instances of this class to be copy constructed */
    IWeightsManager(const IWeightsManager &) = delete;
    /** Prevent instances of this class to be copied */
    IWeightsManager &operator=(const IWeightsManager &) = delete;
    /** Allow instances of this class to be move constructed */
    IWeightsManager(IWeightsManager &&) = default;
    /** Allow instances of this class to be moved */
    IWeightsManager &operator=(IWeightsManager &&) = default;

    /** Start managing a weights tensor
     *
     * @param[in] weights Pointer to the weights tensor to be managed
     * @param[in] parent  Parent node in case where the weights are coming from a previous reshape function
     */
    void manage(const ITensor *weights, ITransformWeights *parent = nullptr);
    /** Run the reshape function.
     *
     * @param[in] weights           Pointer to the weights tensor we want to reshape
     * @param[in] weights_transform Weights transformation object
     *
     * @return The reshaped tensor
     */
    ITensor *run(const ITensor *weights, ITransformWeights *weights_transform);
    /** Acquire the requested reshape tensor of the selected weights
     *
     * @param[in] weights           Pointer to the weights tensor to be managed
     * @param[in] weights_transform Weights transformation object
     */
    ITensor *acquire(const ITensor *weights, ITransformWeights *weights_transform);
    /** Check if the weights are managed
     *
     * @param[in] weights Pointer to the weights tensor we want to check if managed
     *
     * @return True if the weights tensor is managed else false
     */
    bool are_weights_managed(const ITensor *weights);
    /** Release weights refcount and mark as unused if reaches 0
     *
     * @param[in] weights Weights to release
     */
    void release(const ITensor *weights);
    /** Marks weights as unused
     *
     * @param weights Weights to mark unused
     */
    void mark_as_unused(const ITensor *weights);

private:
    struct CounterElement
    {
        bool             is_unused{ false };
        std::atomic<int> counter{ 1 };
    };

private:
    std::map<const ITensor *, std::vector<ITransformWeights *>> _managed_weights;
    std::map<const ITensor *, CounterElement>                   _managed_counter;
    std::map<const ITensor *, ITransformWeights *>              _managed_weights_parents;
};
} // arm_compute
#endif /*ARM_COMPUTE_IWEIGHTSMANAGER_H */