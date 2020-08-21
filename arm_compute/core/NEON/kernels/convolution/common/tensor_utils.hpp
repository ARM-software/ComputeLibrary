/*
 * Copyright (c) 2017 Arm Limited.
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

#pragma once
#include "tensor.hpp"

// Methods to print tensors and weights
void PrintTensor(const Tensor4D<Tensor4DShape, float>& tensor);
void PrintWeights(const Tensor4D<KernelShape, float>& weights);

// Test the equivalence of two tensors
// Counts the instances that |a - b|/|a| > max_err
bool CmpTensors(
  const Tensor4D<Tensor4DShape, float>& a,
  const Tensor4D<Tensor4DShape, float>& b,
  const float max_err=0.0f
);

// Fill the tensor with a test pattern
void TestPattern(Tensor4D<Tensor4DShape, float>& tensor);
void TestPattern(Tensor4D<KernelShape, float>& weights);

// Fill the tensor with random values
void Randomise(Tensor4D<Tensor4DShape, float>& tensor, const int seed=0);
void Randomise(Tensor4D<KernelShape, float>& weights, const int seed=0);
