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
#ifndef __ARM_COMPUTE_TEST_VALIDATION_DATASETS_H__
#define __ARM_COMPUTE_TEST_VALIDATION_DATASETS_H__

#include "dataset/ActivationFunctionDataset.h"
#include "dataset/BatchNormalizationLayerDataset.h"
#include "dataset/BorderModeDataset.h"
#include "dataset/ConvertPolicyDataset.h"
#include "dataset/ConvolutionLayerDataset.h"
#include "dataset/DataTypeDatasets.h"
#include "dataset/FullyConnectedLayerDataset.h"
#include "dataset/GEMMDataset.h"
#include "dataset/ImageDatasets.h"
#include "dataset/InterpolationPolicyDataset.h"
#include "dataset/MatrixPatternDataset.h"
#include "dataset/NonLinearFilterFunctionDataset.h"
#include "dataset/NormalizationTypeDataset.h"
#include "dataset/PoolingLayerDataset.h"
#include "dataset/PoolingTypesDataset.h"
#include "dataset/RoundingPolicyDataset.h"
#include "dataset/ShapeDatasets.h"
#include "dataset/ThresholdDataset.h"

#include "boost_wrapper.h"

using namespace boost::unit_test::data::monomorphic;

namespace boost
{
namespace unit_test
{
namespace data
{
namespace monomorphic
{
/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::SmallImages> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::LargeImages> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::SmallShapes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::Small1DShape> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::Small2DShapes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::LargeShapes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::Large2DShapes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::AllDataTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::UnsignedDataTypes> : boost::mpl::true_
{
};

// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::SignedDataTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::FloatDataTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::FixedPointDataTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::CNNFloatDataTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::CNNFixedPointDataTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::CNNDataTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::ActivationFunctions> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::BorderModes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::ConvertPolicies> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::InterpolationPolicies> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::NormalizationTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::RandomPoolingLayerDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::RoundingPolicies> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::PoolingTypes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::AlexNetConvolutionLayerDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::AlexNetFullyConnectedLayerDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::DirectConvolutionShapes> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::SmallFullyConnectedLayerDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::LargeFullyConnectedLayerDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::SmallConvolutionLayerDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::SmallGEMMDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::LargeGEMMDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::RandomBatchNormalizationLayerDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::ThresholdDataset> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::NonLinearFilterFunctions> : boost::mpl::true_
{
};

/// Register the data set with Boost
template <>
struct is_dataset<arm_compute::test::MatrixPatterns> : boost::mpl::true_
{
};
}
}
}
}
#endif
