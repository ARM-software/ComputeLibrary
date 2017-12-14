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
#include <cstdio>
#include <cstdlib>
#include <random>

#include "alloc.hpp"

/*****************************************************************************/
/* Padding definitions */
enum PaddingType {
  PADDING_SAME, PADDING_VALID
};

/*****************************************************************************/
/* Shape of a kernel */
struct KernelShape {
  int n_output_channels, n_rows, n_cols, n_input_channels;

  int size(void) const {
    return n_output_channels * n_rows * n_cols * n_input_channels;
  }
};

struct Tensor4DShape {
  int n_batches,
      n_rows,
      n_cols,
      n_channels;

  int size() const {
    return n_batches * n_rows * n_cols * n_channels;
  }

  bool TestEq(const Tensor4DShape& other) const {
    return (n_batches == other.n_batches &&
            n_rows == other.n_rows &&
            n_cols == other.n_cols &&
            n_channels == other.n_channels);
  }
};

template <typename ShapeT, typename T>
class Tensor4D final {
  public:
    Tensor4D(ShapeT shape) :
      _shape(shape),
      _data(reinterpret_cast<T*>(ALLOCATE(size_bytes()))) {
        Clear();
    }

    ~Tensor4D() {
      free(_data);
    }

    T* ptr() const {
      return _data;
    }

    const ShapeT& shape() const {
      return _shape;
    }

    size_t size_bytes() const {
      return _shape.size() * sizeof(T);
    }

    bool TestEq(Tensor4D<ShapeT, T>& other) const;
    T& element(int, int, int, int) const;
    void Print() const;

    void Clear() {
      Fill(static_cast<T>(0));
    }

    void Fill(T val) {
      for (int i = 0; i < _shape.size(); i++)
        _data[i] = val;
    }

    void TestPattern() {
      for (int i = 0; i < _shape.size(); i++)
        _data[i] = static_cast<T>(i);
    }

    void Rand(const int seed=2311) {
      std::mt19937 gen(seed);
      std::uniform_int_distribution<> dis(-50, +50);

      for (int i = 0; i < _shape.size(); i++) {
        _data[i] = static_cast<T>(dis(gen));
      }
    }
    Tensor4D(const Tensor4D &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Tensor4D &operator=(const Tensor4D &) = delete;
    /** Allow instances of this class to be moved */
    Tensor4D(Tensor4D &&) = default;
    /** Allow instances of this class to be moved */
    Tensor4D &operator=(Tensor4D &&) = default;


  private:
    const ShapeT _shape;
    T* const _data;
};


template <>
inline float& Tensor4D<Tensor4DShape, float>::element(int n, int i, int j, int c) const {
  int index = ((n*_shape.n_rows + i)*_shape.n_cols + j)*_shape.n_channels + c;
  return _data[index];
}


template <>
inline float& Tensor4D<KernelShape, float>::element(int oc, int i, int j, int ic) const {
  int index = ((i*_shape.n_cols + j)*_shape.n_input_channels + ic)*_shape.n_output_channels + oc;
  return _data[index];
}

template <>
inline bool Tensor4D<Tensor4DShape, float>::TestEq(Tensor4D<Tensor4DShape, float>& other) const {
  // Test equivalence, printing errors
  // First test the shapes are the same
  if (!_shape.TestEq(other.shape())) {
    printf("Tensors have different shapes.\n");
    return false;
  } else {
    int incorrects = 0;

    for (int n = 0; n < _shape.n_batches; n++) {
      for (int i = 0; i < _shape.n_rows; i++) {
        for (int j = 0; j < _shape.n_cols; j++) {
          for (int c = 0; c < _shape.n_channels; c++) {
            // Check elements for equivalence
            const auto a = this->element(n, i, j, c);
            const auto b = other.element(n, i, j, c);

            if (a != b) {
              printf("Difference at element {%d, %d, %d, %d}: %.3f != %.3f\n", n, i, j, c, a, b);

              if (++incorrects > 100) {
                printf("More than 100 incorrect values, stopping test.\n");
                return false;
              }
            }
          }
        }
      }
    }

    return incorrects == 0;
  }
}


template <>
inline void Tensor4D<Tensor4DShape, float>::Print() const {
  for (int n = 0; n < _shape.n_batches; n++) {
    for (int c = 0; c < _shape.n_channels; c++) {
      for (int i = 0; i < _shape.n_rows; i++) {
        for (int j = 0; j < _shape.n_cols; j++) {
          printf("%5.2f ", element(n, i, j, c));
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}


template <>
inline void Tensor4D<KernelShape, float>::Print() const {
  for (int oc = 0; oc < _shape.n_output_channels; oc++) {
    for (int ic = 0; ic < _shape.n_input_channels; ic++) {
      for (int i = 0; i < _shape.n_rows; i++) {
        for (int j = 0; j < _shape.n_cols; j++) {
          printf("%5.2f ", element(oc, i, j, ic));
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}
