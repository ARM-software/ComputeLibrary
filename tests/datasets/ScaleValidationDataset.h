/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_SCALE_VALIDATION_DATASET
#define ARM_COMPUTE_TEST_SCALE_VALIDATION_DATASET

#include "utils/TypePrinter.h"

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/datasets/BorderModeDataset.h"
#include "tests/datasets/InterpolationPolicyDataset.h"
#include "tests/datasets/SamplingPolicyDataset.h"
#include "tests/datasets/ShapeDatasets.h"

namespace arm_compute
{
namespace test
{
namespace datasets
{
/** Class to generate boundary values for the given template parameters
 * including shapes with large differences between width and height.
 * element_per_iteration is the number of elements processed by one iteration
 * of an implementation. (E.g., if an iteration is based on a 16-byte vector
 * and size of one element is 1-byte, this value would be 16.).
 * iterations is the total number of complete iterations we want to test
 * for the effect of larger shapes.
 */
template <uint32_t channel, uint32_t batch, uint32_t element_per_iteration, uint32_t iterations>
class ScaleShapesBaseDataSet : public ShapeDataset
{
    static constexpr auto boundary_minus_one = element_per_iteration * iterations - 1;
    static constexpr auto boundary_plus_one  = element_per_iteration * iterations + 1;
    static constexpr auto small_size         = 3;

public:
    // These tensor shapes are NCHW layout, fixture will convert to NHWC.
    ScaleShapesBaseDataSet()
        : ShapeDataset("Shape",
    {
        TensorShape{ small_size, boundary_minus_one, channel, batch },
                     TensorShape{ small_size, boundary_plus_one, channel, batch },
                     TensorShape{ boundary_minus_one, small_size, channel, batch },
                     TensorShape{ boundary_plus_one, small_size, channel, batch },
                     TensorShape{ boundary_minus_one, boundary_plus_one, channel, batch },
                     TensorShape{ boundary_plus_one, boundary_minus_one, channel, batch },
    })
    {
    }
};

/** For the single vector, only larger value (+1) than boundary
 * since smaller value (-1) could cause some invalid shapes like
 * - invalid zero size
 * - size 1 which isn't compatible with scale with aligned corners.
 */
template <uint32_t channel, uint32_t batch, uint32_t element_per_iteration>
class ScaleShapesBaseDataSet<channel, batch, element_per_iteration, 1> : public ShapeDataset
{
    static constexpr auto small_size        = 3;
    static constexpr auto boundary_plus_one = element_per_iteration + 1;

public:
    // These tensor shapes are NCHW layout, fixture will convert to NHWC.
    ScaleShapesBaseDataSet()
        : ShapeDataset("Shape",
    {
        TensorShape{ small_size, boundary_plus_one, channel, batch },
                     TensorShape{ boundary_plus_one, small_size, channel, batch },
    })
    {
    }
};

/** For the shapes smaller than one vector, only pre-defined tiny shapes
 * are tested (3x2, 2x3) as smaller shapes are more likely to cause
 * issues and easier to debug.
 */
template <uint32_t channel, uint32_t batch, uint32_t element_per_iteration>
class ScaleShapesBaseDataSet<channel, batch, element_per_iteration, 0> : public ShapeDataset
{
    static constexpr auto small_size                 = 3;
    static constexpr auto zero_vector_boundary_value = 2;

public:
    // These tensor shapes are NCHW layout, fixture will convert to NHWC.
    ScaleShapesBaseDataSet()
        : ShapeDataset("Shape",
    {
        TensorShape{ small_size, zero_vector_boundary_value, channel, batch },
                     TensorShape{ zero_vector_boundary_value, small_size, channel, batch },
    })
    {
    }
};

/** Interpolation policy test set */
const auto ScaleInterpolationPolicySet = framework::dataset::make("InterpolationPolicy",
{
    InterpolationPolicy::NEAREST_NEIGHBOR,
    InterpolationPolicy::BILINEAR,
});

/** Scale data types */
const auto ScaleDataLayouts = framework::dataset::make("DataLayout",
{
    DataLayout::NCHW,
    DataLayout::NHWC,
});

/** Sampling policy data set */
const auto ScaleSamplingPolicySet = combine(datasets::SamplingPolicies(),
                                            framework::dataset::make("AlignCorners", { false }));

/** Sampling policy data set for Aligned Corners which only allows TOP_LEFT policy.*/
const auto ScaleAlignCornersSamplingPolicySet = combine(framework::dataset::make("SamplingPolicy",
{
    SamplingPolicy::TOP_LEFT,
}),
framework::dataset::make("AlignCorners", { true }));

/** Generated shapes: Used by Neon precommit and nightly
 * - 2D shapes with 0, 1, 2 vector iterations
 * - 3D shapes with 0, 1 vector iterations
 * - 4D shapes with 0 vector iterations
 */
#define SCALE_SHAPE_DATASET(element_per_iteration)                                                  \
    concat(concat(concat(concat(concat(ScaleShapesBaseDataSet<1, 1, (element_per_iteration), 0>(),  \
                                       ScaleShapesBaseDataSet<1, 1, (element_per_iteration), 1>()), \
                                ScaleShapesBaseDataSet<1, 1, (element_per_iteration), 2>()),        \
                         ScaleShapesBaseDataSet<3, 1, (element_per_iteration), 0>()),               \
                  ScaleShapesBaseDataSet<3, 1, (element_per_iteration), 1>()),                      \
           ScaleShapesBaseDataSet<3, 3, (element_per_iteration), 0>())

// To prevent long precommit time for OpenCL, shape set for OpenCL is separated into below two parts.
/** Generated shapes for precommits to achieve essential coverage. Used by CL precommit and nightly
 * - 3D shapes with 1 vector iterations
 * - 4D shapes with 1 vector iterations
 */
#define SCALE_PRECOMMIT_SHAPE_DATASET(element_per_iteration) \
    concat(ScaleShapesBaseDataSet<3, 1, (element_per_iteration), 1>(), ScaleShapesBaseDataSet<3, 3, (element_per_iteration), 1>())

/** Generated shapes for nightly to achieve more small and variety shapes. Used by CL nightly
 * - 2D shapes with 0, 1, 2 vector iterations
 * - 3D shapes with 0 vector iterations (1 vector iteration is covered by SCALE_PRECOMMIT_SHAPE_DATASET)
 * - 4D shapes with 0 vector iterations
 */
#define SCALE_NIGHTLY_SHAPE_DATASET(element_per_iteration)                                   \
    concat(concat(concat(concat(ScaleShapesBaseDataSet<1, 1, (element_per_iteration), 0>(),  \
                                ScaleShapesBaseDataSet<1, 1, (element_per_iteration), 1>()), \
                         ScaleShapesBaseDataSet<1, 1, (element_per_iteration), 2>()),        \
                  ScaleShapesBaseDataSet<3, 1, (element_per_iteration), 0>()),               \
           ScaleShapesBaseDataSet<3, 3, (element_per_iteration), 0>())

/** Generating dataset for non-quantized data tyeps with the given shapes */
#define ASSEMBLE_DATASET(shape, samping_policy_set)             \
    combine(combine(combine(combine((shape), ScaleDataLayouts), \
                            ScaleInterpolationPolicySet),       \
                    datasets::BorderModes()),                   \
            samping_policy_set)

/** Generating dataset for quantized data tyeps with the given shapes */
#define ASSEMBLE_QUANTIZED_DATASET(shape, sampling_policy_set, quantization_info_set) \
    combine(combine(combine(combine(combine(shape,                                    \
                                            quantization_info_set),                   \
                                    ScaleDataLayouts),                                \
                            ScaleInterpolationPolicySet),                             \
                    datasets::BorderModes()),                                         \
            sampling_policy_set)

} // namespace datasets
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALE_VALIDATION_DATASET */
