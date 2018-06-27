/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_UTILS_H__
#define __ARM_COMPUTE_GRAPH_UTILS_H__

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include <array>
#include <random>
#include <string>
#include <vector>

namespace arm_compute
{
namespace graph_utils
{
/** Preprocessor interface **/
class IPreprocessor
{
public:
    /** Default destructor. */
    virtual ~IPreprocessor() = default;
    /** Preprocess the given tensor.
     *
     * @param[in] tensor Tensor to preprocess.
     */
    virtual void preprocess(ITensor &tensor) = 0;
};

/** Caffe preproccessor */
class CaffePreproccessor : public IPreprocessor
{
public:
    /** Default Constructor
     *
     * @param mean Mean array in RGB ordering
     * @param bgr  Boolean specifying if the preprocessing should assume BGR format
     */
    CaffePreproccessor(std::array<float, 3> mean = std::array<float, 3> { { 0, 0, 0 } }, bool bgr = true);
    void preprocess(ITensor &tensor) override;

private:
    std::array<float, 3> _mean;
    bool _bgr;
};

/** TF preproccessor */
class TFPreproccessor : public IPreprocessor
{
public:
    void preprocess(ITensor &tensor) override;
};

/** PPM writer class */
class PPMWriter : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] name    PPM file name
     * @param[in] maximum Maximum elements to access
     */
    PPMWriter(std::string name, unsigned int maximum = 1);
    /** Allows instances to move constructed */
    PPMWriter(PPMWriter &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string _name;
    unsigned int      _iterator;
    unsigned int      _maximum;
};

/** Dummy accessor class */
class DummyAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] maximum Maximum elements to write
     */
    DummyAccessor(unsigned int maximum = 1);
    /** Allows instances to move constructed */
    DummyAccessor(DummyAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    unsigned int _iterator;
    unsigned int _maximum;
};

/** NumPy accessor class */
class NumPyAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in]  npy_path      Path to npy file.
     * @param[in]  shape         Shape of the numpy tensor data.
     * @param[in]  data_type     DataType of the numpy tensor data.
     * @param[out] output_stream (Optional) Output stream
     */
    NumPyAccessor(std::string npy_path, TensorShape shape, DataType data_type, std::ostream &output_stream = std::cout);
    /** Allow instances of this class to be move constructed */
    NumPyAccessor(NumPyAccessor &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NumPyAccessor(const NumPyAccessor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NumPyAccessor &operator=(const NumPyAccessor &) = delete;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    template <typename T>
    void access_numpy_tensor(ITensor &tensor);

    Tensor            _npy_tensor;
    const std::string _filename;
    std::ostream     &_output_stream;
};

/** PPM accessor class */
class PPMAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] ppm_path     Path to PPM file
     * @param[in] bgr          (Optional) Fill the first plane with blue channel (default = false - RGB format)
     * @param[in] preprocessor (Optional) PPM pre-processing object
     */
    PPMAccessor(std::string ppm_path, bool bgr = true, std::unique_ptr<IPreprocessor> preprocessor = nullptr);
    /** Allow instances of this class to be move constructed */
    PPMAccessor(PPMAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string              _ppm_path;
    const bool                     _bgr;
    std::unique_ptr<IPreprocessor> _preprocessor;
};

/** Input Accessor used for network validation */
class ValidationInputAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] image_list  File containing all the images to validate
     * @param[in] images_path Path to images.
     * @param[in] bgr         (Optional) Fill the first plane with blue channel (default = false - RGB format)
     * @param[in] start       (Optional) Start range
     * @param[in] end         (Optional) End range
     *
     * @note Range is defined as [start, end]
     */
    ValidationInputAccessor(const std::string &image_list,
                            std::string        images_path,
                            bool               bgr   = true,
                            unsigned int       start = 0,
                            unsigned int       end   = 0);

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    std::string              _path;
    std::vector<std::string> _images;
    bool                     _bgr;
    size_t                   _offset;
};

/** Output Accessor used for network validation */
class ValidationOutputAccessor final : public graph::ITensorAccessor
{
public:
    /** Default Constructor
     *
     * @param[in]  image_list    File containing all the images and labels results
     * @param[in]  top_n         (Optional) Top N accuracy (Defaults to 5)
     * @param[out] output_stream (Optional) Output stream (Defaults to the standard output stream)
     * @param[in]  start         (Optional) Start range
     * @param[in]  end           (Optional) End range
     *
     * @note Range is defined as [start, end]
     */
    ValidationOutputAccessor(const std::string &image_list,
                             size_t             top_n         = 5,
                             std::ostream      &output_stream = std::cout,
                             unsigned int       start         = 0,
                             unsigned int       end           = 0);
    /** Reset accessor state */
    void reset();

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    /** Access predictions of the tensor
     *
     * @tparam T Tensor elements type
     *
     * @param[in] tensor Tensor to read the predictions from
     */
    template <typename T>
    std::vector<size_t> access_predictions_tensor(ITensor &tensor);

private:
    std::vector<int> _results;
    std::ostream    &_output_stream;
    size_t           _top_n;
    size_t           _offset;
    size_t           _positive_samples;
};

/** Result accessor class */
class TopNPredictionsAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in]  labels_path   Path to labels text file.
     * @param[in]  top_n         (Optional) Number of output classes to print
     * @param[out] output_stream (Optional) Output stream
     */
    TopNPredictionsAccessor(const std::string &labels_path, size_t top_n = 5, std::ostream &output_stream = std::cout);
    /** Allow instances of this class to be move constructed */
    TopNPredictionsAccessor(TopNPredictionsAccessor &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    TopNPredictionsAccessor(const TopNPredictionsAccessor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    TopNPredictionsAccessor &operator=(const TopNPredictionsAccessor &) = delete;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    template <typename T>
    void access_predictions_tensor(ITensor &tensor);

    std::vector<std::string> _labels;
    std::ostream            &_output_stream;
    size_t                   _top_n;
};

/** Random accessor class */
class RandomAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] lower Lower bound value.
     * @param[in] upper Upper bound value.
     * @param[in] seed  (Optional) Seed used to initialise the random number generator.
     */
    RandomAccessor(PixelValue lower, PixelValue upper, const std::random_device::result_type seed = 0);
    /** Allows instances to move constructed */
    RandomAccessor(RandomAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    template <typename T, typename D>
    void fill(ITensor &tensor, D &&distribution);
    PixelValue                      _lower;
    PixelValue                      _upper;
    std::random_device::result_type _seed;
};

/** Numpy Binary loader class*/
class NumPyBinLoader final : public graph::ITensorAccessor
{
public:
    /** Default Constructor
     *
     * @param[in] filename    Binary file name
     * @param[in] file_layout (Optional) Layout of the numpy tensor data. Defaults to NCHW
     */
    NumPyBinLoader(std::string filename, DataLayout file_layout = DataLayout::NCHW);
    /** Allows instances to move constructed */
    NumPyBinLoader(NumPyBinLoader &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string _filename;
    const DataLayout  _file_layout;
};

/** Generates appropriate random accessor
 *
 * @param[in] lower Lower random values bound
 * @param[in] upper Upper random values bound
 * @param[in] seed  Random generator seed
 *
 * @return A ramdom accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_random_accessor(PixelValue lower, PixelValue upper, const std::random_device::result_type seed = 0)
{
    return arm_compute::support::cpp14::make_unique<RandomAccessor>(lower, upper, seed);
}

/** Generates appropriate weights accessor according to the specified path
 *
 * @note If path is empty will generate a DummyAccessor else will generate a NumPyBinLoader
 *
 * @param[in] path        Path to the data files
 * @param[in] data_file   Relative path to the data files from path
 * @param[in] file_layout (Optional) Layout of file. Defaults to NCHW
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_weights_accessor(const std::string &path,
                                                                    const std::string &data_file,
                                                                    DataLayout         file_layout = DataLayout::NCHW)
{
    if(path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<NumPyBinLoader>(path + data_file, file_layout);
    }
}

/** Generates appropriate input accessor according to the specified ppm_path
 *
 * @note If ppm_path is empty will generate a DummyAccessor else will generate a PPMAccessor
 *
 * @param[in] ppm_path     Path to PPM file
 * @param[in] preprocessor Preproccessor object
 * @param[in] bgr          (Optional) Fill the first plane with blue channel (default = true)
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_input_accessor(const std::string             &ppm_path,
                                                                  std::unique_ptr<IPreprocessor> preprocessor = nullptr,
                                                                  bool                           bgr          = true)
{
    if(ppm_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>();
    }
    else
    {
        if(arm_compute::utility::endswith(ppm_path, ".npy"))
        {
            return arm_compute::support::cpp14::make_unique<NumPyBinLoader>(ppm_path);
        }
        else
        {
            return arm_compute::support::cpp14::make_unique<PPMAccessor>(ppm_path, bgr, std::move(preprocessor));
        }
    }
}

/** Generates appropriate output accessor according to the specified labels_path
 *
 * @note If labels_path is empty will generate a DummyAccessor else will generate a TopNPredictionsAccessor
 *
 * @param[in]  labels_path   Path to labels text file
 * @param[in]  top_n         (Optional) Number of output classes to print
 * @param[out] output_stream (Optional) Output stream
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_output_accessor(const std::string &labels_path, size_t top_n = 5, std::ostream &output_stream = std::cout)
{
    if(labels_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>(0);
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<TopNPredictionsAccessor>(labels_path, top_n, output_stream);
    }
}
/** Generates appropriate npy output accessor according to the specified npy_path
 *
 * @note If npy_path is empty will generate a DummyAccessor else will generate a NpyAccessor
 *
 * @param[in]  npy_path      Path to npy file.
 * @param[in]  shape         Shape of the numpy tensor data.
 * @param[in]  data_type     DataType of the numpy tensor data.
 * @param[out] output_stream (Optional) Output stream
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_npy_output_accessor(const std::string &npy_path, TensorShape shape, DataType data_type, std::ostream &output_stream = std::cout)
{
    if(npy_path.empty())
    {
        return arm_compute::support::cpp14::make_unique<DummyAccessor>(0);
    }
    else
    {
        return arm_compute::support::cpp14::make_unique<NumPyAccessor>(npy_path, shape, data_type, output_stream);
    }
}

/** Utility function to return the TargetHint
 *
 * @param[in] target Integer value which expresses the selected target. Must be 0 for NEON or 1 for OpenCL or 2 (OpenCL with Tuner)
 *
 * @return the TargetHint
 */
inline graph::Target set_target_hint(int target)
{
    ARM_COMPUTE_ERROR_ON_MSG(target > 3, "Invalid target. Target must be 0 (NEON), 1 (OpenCL), 2 (OpenCL + Tuner), 3 (GLES)");
    if((target == 1 || target == 2))
    {
        return graph::Target::CL;
    }
    else if(target == 3)
    {
        return graph::Target::GC;
    }
    else
    {
        return graph::Target::NEON;
    }
}
} // namespace graph_utils
} // namespace arm_compute

#endif /* __ARM_COMPUTE_GRAPH_UTILS_H__ */
