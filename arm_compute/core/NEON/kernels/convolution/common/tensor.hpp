/*
 * Copyright (c) 2017 ARM Limited.
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
#include <cstdlib>
#include <random>

#include "alloc.hpp"

enum TensorOrder
{
  NHWC,  ///< [Batch x Height x Width x Channels]
  NCHW,  ///< [Batch x Channels x Height x Width]
};

struct Tensor4DShape
{
  int n_batches, n_rows, n_cols, n_channels;
  TensorOrder ordering;

  // Create a new tensor with the default (NHWC) ordering
  inline Tensor4DShape(
    const int n_batches,
    const int n_rows,
    const int n_cols,
    const int n_channels,
    const TensorOrder ordering=NHWC
  ) : n_batches(n_batches),
      n_rows(n_rows),
      n_cols(n_cols),
      n_channels(n_channels),
      ordering(ordering)
  {
  }

  inline int size() const
  {
    return n_batches * n_rows * n_cols * n_channels;
  }

  inline bool TestEq(const Tensor4DShape& other) const
  {
    return (n_batches == other.n_batches &&
            n_rows == other.n_rows &&
            n_cols == other.n_cols &&
            n_channels == other.n_channels);
  }
};


enum WeightOrder
{
  HWIO,  ///< [Height x Width x Input channels x Output channels]
  OIHW,  ///< [Output channels x Input channels x Height x Width]
};

struct KernelShape
{
  int n_output_channels, n_rows, n_cols, n_input_channels;
  WeightOrder ordering;

  inline KernelShape(
    const int n_output_channels,
    const int n_rows,
    const int n_cols,
    const int n_input_channels,
    const WeightOrder ordering=HWIO
  ) : n_output_channels(n_output_channels),
      n_rows(n_rows),
      n_cols(n_cols),
      n_input_channels(n_input_channels),
      ordering(ordering)
  {
  }

  inline int size(void) const
  {
    return n_output_channels * n_rows * n_cols * n_input_channels;
  }
};


template <typename ShapeT, typename T>
class Tensor4D final
{
  public:
    Tensor4D(ShapeT shape) :
      shape(shape),
      _data(reinterpret_cast<T*>(ALLOCATE(size_bytes())))
    {
        Clear();
    }

    Tensor4D(const Tensor4D<ShapeT, T>&) = delete;
    Tensor4D operator=(const Tensor4D<ShapeT, T>&) = delete;

    ~Tensor4D() {
      free(_data);
    }

    inline T* ptr() const {
      return _data;
    }

    inline size_t size_bytes() const {
      return shape.size() * sizeof(T);
    }

    inline T& element(int, int, int, int) const;

    inline void Clear() {
      Fill(static_cast<T>(0));
    }

    inline void Fill(T val) {
      for (int i = 0; i < shape.size(); i++)
        _data[i] = val;
    }

    const ShapeT shape;

  private:
    T* const _data;
};


template <>
inline float& Tensor4D<Tensor4DShape, float>::element(int n, int i, int j, int c) const
{
  int index;
  if (shape.ordering == NHWC)
  {
    index = ((n*shape.n_rows + i)*shape.n_cols + j)*shape.n_channels + c;
  }
  else  // NCHW
  {
    index = ((n*shape.n_channels + c)*shape.n_rows + i)*shape.n_cols + j;
  }
  return _data[index];
}


template <>
inline float& Tensor4D<KernelShape, float>::element(int oc, int i, int j, int ic) const
{
  int index;
  if (shape.ordering == HWIO)
  {
    index = ((i*shape.n_cols + j)*shape.n_input_channels + ic)*shape.n_output_channels + oc;
  }
  else  // OIHW
  {
    index = ((oc*shape.n_input_channels + ic)*shape.n_rows + i)*shape.n_cols + j;
  }
  return _data[index];
}
