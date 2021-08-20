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
#include "arm_compute/runtime/IWeightsManager.h"

namespace arm_compute
{
IWeightsManager::IWeightsManager()
    : _managed_weights(), _managed_counter(), _managed_weights_parents()
{
}

void IWeightsManager::manage(const ITensor *weights, ITransformWeights *parent)
{
    if(!are_weights_managed(weights))
    {
        _managed_weights[weights];
        _managed_counter[weights];
    }
    else
    {
        _managed_counter[weights].counter++;
    }

    // In case the weights are an output of a previous reshape function
    // store the parent's link
    if(parent != nullptr)
    {
        if(_managed_weights_parents.find(weights) == _managed_weights_parents.end())
        {
            _managed_weights_parents[weights] = parent;
        }
    }
}

ITensor *IWeightsManager::run(const ITensor *weights, ITransformWeights *weights_transform)
{
    ARM_COMPUTE_ERROR_ON_MSG(!are_weights_managed(weights), "Cannot run function. Weights are not managed");

    // Find if I have the same weights with weights transform. If I do, don't run the reshape
    auto     item = _managed_weights.find(weights);
    bool     perform_run{ true };
    ITensor *weights_tensor{ nullptr };

    // Check if I already have the requested transform and I have run the reshape function
    for(auto it : item->second)
    {
        if(it->is_reshape_run() && (it->uid() == weights_transform->uid()))
        {
            weights_tensor = it->get_weights();
            perform_run    = false;
            break;
        }
    }

    if(perform_run)
    {
        weights_transform->run();
        weights_tensor = weights_transform->get_weights();
    }

    // Check if we can release memory from parent
    auto parent_item = _managed_weights_parents.find(weights);
    if(parent_item != _managed_weights_parents.end())
    {
        int32_t refcount = parent_item->second->decrease_refcount();
        if(refcount == 0)
        {
            parent_item->second->release();
        }
    }

    // Check top level weights. If all the transformations are done
    // mark the weights as unused
    if(_managed_weights_parents.find(weights) == _managed_weights_parents.end())
    {
        auto item           = _managed_weights.find(weights);
        bool mark_as_unused = true;
        for(auto it : item->second)
        {
            if(!it->is_reshape_run())
            {
                mark_as_unused = false;
                break;
            }
        }

        if(mark_as_unused)
        {
            weights->mark_as_unused();
        }
    }

    return weights_tensor;
}

bool IWeightsManager::are_weights_managed(const ITensor *weights)
{
    return (_managed_weights.find(weights) != _managed_weights.end());
}

ITensor *IWeightsManager::acquire(const ITensor *weights, ITransformWeights *weights_transform)
{
    ARM_COMPUTE_ERROR_ON_MSG(!are_weights_managed(weights), "Cannot acquire weights. Weights are not managed");

    ITensor *transformed_weights{ nullptr };
    auto     item = _managed_weights.find(weights);

    // Check if I already have the requested transform. If I do,
    // increase the refcount of the transformed weights object and
    // reuse the tensor
    for(auto it : item->second)
    {
        if(it->uid() == weights_transform->uid())
        {
            transformed_weights = it->get_weights();
            it->increase_refcount();
            break;
        }
    }

    if(transformed_weights == nullptr)
    {
        transformed_weights = weights_transform->get_weights();
        weights_transform->increase_refcount();
        item->second.emplace_back(weights_transform);
    }

    // Manage the weights and store link to the parent node
    manage(transformed_weights, weights_transform);

    return transformed_weights;
}

void IWeightsManager::release(const ITensor *weights)
{
    if(weights == nullptr || !are_weights_managed(weights))
    {
        return;
    }

    _managed_counter[weights].counter--;
    if(_managed_counter[weights].counter == 0 && _managed_counter[weights].is_unused)
    {
        weights->mark_as_unused();
    }
}

void IWeightsManager::mark_as_unused(const ITensor *weights)
{
    if(weights == nullptr || !are_weights_managed(weights))
    {
        return;
    }

    _managed_counter[weights].is_unused = true;
}
} // namespace arm_compute
