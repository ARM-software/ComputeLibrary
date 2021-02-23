/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef __ARM_COMPUTE_UTILS_GRAPH_UTILS_H__
#define __ARM_COMPUTE_UTILS_GRAPH_UTILS_H__

#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/Utility.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/ITensorAccessor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/Tensor.h"

#include "utils/CommonGraphOptions.h"

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
     * @param[in] mean  Mean array in RGB ordering
     * @param[in] bgr   Boolean specifying if the preprocessing should assume BGR format
     * @param[in] scale Scale value
     */
    CaffePreproccessor(std::array<float, 3> mean = std::array<float, 3> { { 0, 0, 0 } }, bool bgr = true, float scale = 1.f);
    void preprocess(ITensor &tensor) override;

private:
    template <typename T>
    void preprocess_typed(ITensor &tensor);

    std::array<float, 3> _mean;
    bool  _bgr;
    float _scale;
};

/** TF preproccessor */
class TFPreproccessor : public IPreprocessor
{
public:
    /** Constructor
     *
     * @param[in] min_range Min normalization range. (Defaults to -1.f)
     * @param[in] max_range Max normalization range. (Defaults to 1.f)
     */
    TFPreproccessor(float min_range = -1.f, float max_range = 1.f);

    // Inherited overriden methods
    void preprocess(ITensor &tensor) override;

private:
    template <typename T>
    void preprocess_typed(ITensor &tensor);

    float _min_range;
    float _max_range;
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
     * @param[in]  data_layout   (Optional) DataLayout of the numpy tensor data.
     * @param[out] output_stream (Optional) Output stream
     */
    NumPyAccessor(std::string npy_path, TensorShape shape, DataType data_type, DataLayout data_layout = DataLayout::NCHW, std::ostream &output_stream = std::cout);
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
    void access_numpy_tensor(ITensor &tensor, T tolerance);

    Tensor            _npy_tensor;
    const std::string _filename;
    std::ostream     &_output_stream;
};

/** SaveNumPy accessor class */
class SaveNumPyAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] npy_name   Npy file name.
     * @param[in] is_fortran (Optional) If true, save tensor in fortran order.
     */
    SaveNumPyAccessor(const std::string npy_name, const bool is_fortran = false);
    /** Allow instances of this class to be move constructed */
    SaveNumPyAccessor(SaveNumPyAccessor &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    SaveNumPyAccessor(const SaveNumPyAccessor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    SaveNumPyAccessor &operator=(const SaveNumPyAccessor &) = delete;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    const std::string _npy_name;
    const bool        _is_fortran;
};

/** Print accessor class
 *  @note The print accessor will print only when asserts are enabled.
 *  */
class PrintAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[out] output_stream (Optional) Output stream
     * @param[in]  io_fmt        (Optional) Format information
     */
    PrintAccessor(std::ostream &output_stream = std::cout, IOFormatInfo io_fmt = IOFormatInfo());
    /** Allow instances of this class to be move constructed */
    PrintAccessor(PrintAccessor &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    PrintAccessor(const PrintAccessor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    PrintAccessor &operator=(const PrintAccessor &) = delete;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    std::ostream &_output_stream;
    IOFormatInfo  _io_fmt;
};

/** Image accessor class */
class ImageAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in] filename     Image file
     * @param[in] bgr          (Optional) Fill the first plane with blue channel (default = false - RGB format)
     * @param[in] preprocessor (Optional) Image pre-processing object
     */
    ImageAccessor(std::string filename, bool bgr = true, std::unique_ptr<IPreprocessor> preprocessor = nullptr);
    /** Allow instances of this class to be move constructed */
    ImageAccessor(ImageAccessor &&) = default;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    bool                           _already_loaded;
    const std::string              _filename;
    const bool                     _bgr;
    std::unique_ptr<IPreprocessor> _preprocessor;
};

/** Input Accessor used for network validation */
class ValidationInputAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in]  image_list    File containing all the images to validate
     * @param[in]  images_path   Path to images.
     * @param[in]  bgr           (Optional) Fill the first plane with blue channel (default = false - RGB format)
     * @param[in]  preprocessor  (Optional) Image pre-processing object  (default = nullptr)
     * @param[in]  start         (Optional) Start range
     * @param[in]  end           (Optional) End range
     * @param[out] output_stream (Optional) Output stream
     *
     * @note Range is defined as [start, end]
     */
    ValidationInputAccessor(const std::string             &image_list,
                            std::string                    images_path,
                            std::unique_ptr<IPreprocessor> preprocessor  = nullptr,
                            bool                           bgr           = true,
                            unsigned int                   start         = 0,
                            unsigned int                   end           = 0,
                            std::ostream                  &output_stream = std::cout);

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    std::string                    _path;
    std::vector<std::string>       _images;
    std::unique_ptr<IPreprocessor> _preprocessor;
    bool                           _bgr;
    size_t                         _offset;
    std::ostream                  &_output_stream;
};

/** Output Accessor used for network validation */
class ValidationOutputAccessor final : public graph::ITensorAccessor
{
public:
    /** Default Constructor
     *
     * @param[in]  image_list    File containing all the images and labels results
     * @param[out] output_stream (Optional) Output stream (Defaults to the standard output stream)
     * @param[in]  start         (Optional) Start range
     * @param[in]  end           (Optional) End range
     *
     * @note Range is defined as [start, end]
     */
    ValidationOutputAccessor(const std::string &image_list,
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
    /** Aggregates the results of a sample
     *
     * @param[in]     res              Vector containing the results of a graph
     * @param[in,out] positive_samples Positive samples to be updated
     * @param[in]     top_n            Top n accuracy to measure
     * @param[in]     correct_label    Correct label of the current sample
     */
    void aggregate_sample(const std::vector<size_t> &res, size_t &positive_samples, size_t top_n, size_t correct_label);
    /** Reports top N accuracy
     *
     * @param[in] top_n            Top N accuracy that is being reported
     * @param[in] total_samples    Total number of samples
     * @param[in] positive_samples Positive samples
     */
    void report_top_n(size_t top_n, size_t total_samples, size_t positive_samples);

private:
    std::vector<int> _results;
    std::ostream    &_output_stream;
    size_t           _offset;
    size_t           _positive_samples_top1;
    size_t           _positive_samples_top5;
};

/** Detection output accessor class */
class DetectionOutputAccessor final : public graph::ITensorAccessor
{
public:
    /** Constructor
     *
     * @param[in]  labels_path        Path to labels text file.
     * @param[in]  imgs_tensor_shapes Network input images tensor shapes.
     * @param[out] output_stream      (Optional) Output stream
     */
    DetectionOutputAccessor(const std::string &labels_path, std::vector<TensorShape> &imgs_tensor_shapes, std::ostream &output_stream = std::cout);
    /** Allow instances of this class to be move constructed */
    DetectionOutputAccessor(DetectionOutputAccessor &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    DetectionOutputAccessor(const DetectionOutputAccessor &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    DetectionOutputAccessor &operator=(const DetectionOutputAccessor &) = delete;

    // Inherited methods overriden:
    bool access_tensor(ITensor &tensor) override;

private:
    template <typename T>
    void access_predictions_tensor(ITensor &tensor);

    std::vector<std::string> _labels;
    std::vector<TensorShape> _tensor_shapes;
    std::ostream            &_output_stream;
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
    bool              _already_loaded;
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
    return std::make_unique<RandomAccessor>(lower, upper, seed);
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
        return std::make_unique<DummyAccessor>();
    }
    else
    {
        return std::make_unique<NumPyBinLoader>(path + data_file, file_layout);
    }
}

/** Generates appropriate input accessor according to the specified graph parameters
 *
 * @param[in] graph_parameters Graph parameters
 * @param[in] preprocessor     (Optional) Preproccessor object
 * @param[in] bgr              (Optional) Fill the first plane with blue channel (default = true)
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_input_accessor(const arm_compute::utils::CommonGraphParams &graph_parameters,
                                                                  std::unique_ptr<IPreprocessor>               preprocessor = nullptr,
                                                                  bool                                         bgr          = true)
{
    if(!graph_parameters.validation_file.empty())
    {
        return std::make_unique<ValidationInputAccessor>(graph_parameters.validation_file,
                                                         graph_parameters.validation_path,
                                                         std::move(preprocessor),
                                                         bgr,
                                                         graph_parameters.validation_range_start,
                                                         graph_parameters.validation_range_end);
    }
    else
    {
        const std::string &image_file       = graph_parameters.image;
        const std::string &image_file_lower = lower_string(image_file);
        if(arm_compute::utility::endswith(image_file_lower, ".npy"))
        {
            return std::make_unique<NumPyBinLoader>(image_file, graph_parameters.data_layout);
        }
        else if(arm_compute::utility::endswith(image_file_lower, ".jpeg")
                || arm_compute::utility::endswith(image_file_lower, ".jpg")
                || arm_compute::utility::endswith(image_file_lower, ".ppm"))
        {
            return std::make_unique<ImageAccessor>(image_file, bgr, std::move(preprocessor));
        }
        else
        {
            return std::make_unique<DummyAccessor>();
        }
    }
}

/** Generates appropriate output accessor according to the specified graph parameters
 *
 * @note If the output accessor is requested to validate the graph then ValidationOutputAccessor is generated
 *       else if output_accessor_file is empty will generate a DummyAccessor else will generate a TopNPredictionsAccessor
 *
 * @param[in]  graph_parameters Graph parameters
 * @param[in]  top_n            (Optional) Number of output classes to print (default = 5)
 * @param[in]  is_validation    (Optional) Validation flag (default = false)
 * @param[out] output_stream    (Optional) Output stream (default = std::cout)
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_output_accessor(const arm_compute::utils::CommonGraphParams &graph_parameters,
                                                                   size_t                                       top_n         = 5,
                                                                   bool                                         is_validation = false,
                                                                   std::ostream                                &output_stream = std::cout)
{
    ARM_COMPUTE_UNUSED(is_validation);
    if(!graph_parameters.validation_file.empty())
    {
        return std::make_unique<ValidationOutputAccessor>(graph_parameters.validation_file,
                                                          output_stream,
                                                          graph_parameters.validation_range_start,
                                                          graph_parameters.validation_range_end);
    }
    else if(graph_parameters.labels.empty())
    {
        return std::make_unique<DummyAccessor>(0);
    }
    else
    {
        return std::make_unique<TopNPredictionsAccessor>(graph_parameters.labels, top_n, output_stream);
    }
}
/** Generates appropriate output accessor according to the specified graph parameters
 *
 * @note If the output accessor is requested to validate the graph then ValidationOutputAccessor is generated
 *       else if output_accessor_file is empty will generate a DummyAccessor else will generate a TopNPredictionsAccessor
 *
 * @param[in]  graph_parameters Graph parameters
 * @param[in]  tensor_shapes    Network input images tensor shapes.
 * @param[in]  is_validation    (Optional) Validation flag (default = false)
 * @param[out] output_stream    (Optional) Output stream (default = std::cout)
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_detection_output_accessor(const arm_compute::utils::CommonGraphParams &graph_parameters,
                                                                             std::vector<TensorShape>                     tensor_shapes,
                                                                             bool                                         is_validation = false,
                                                                             std::ostream                                &output_stream = std::cout)
{
    ARM_COMPUTE_UNUSED(is_validation);
    if(!graph_parameters.validation_file.empty())
    {
        return std::make_unique<ValidationOutputAccessor>(graph_parameters.validation_file,
                                                          output_stream,
                                                          graph_parameters.validation_range_start,
                                                          graph_parameters.validation_range_end);
    }
    else if(graph_parameters.labels.empty())
    {
        return std::make_unique<DummyAccessor>(0);
    }
    else
    {
        return std::make_unique<DetectionOutputAccessor>(graph_parameters.labels, tensor_shapes, output_stream);
    }
}
/** Generates appropriate npy output accessor according to the specified npy_path
 *
 * @note If npy_path is empty will generate a DummyAccessor else will generate a NpyAccessor
 *
 * @param[in]  npy_path      Path to npy file.
 * @param[in]  shape         Shape of the numpy tensor data.
 * @param[in]  data_type     DataType of the numpy tensor data.
 * @param[in]  data_layout   DataLayout of the numpy tensor data.
 * @param[out] output_stream (Optional) Output stream
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_npy_output_accessor(const std::string &npy_path, TensorShape shape, DataType data_type, DataLayout data_layout = DataLayout::NCHW,
                                                                       std::ostream &output_stream = std::cout)
{
    if(npy_path.empty())
    {
        return std::make_unique<DummyAccessor>(0);
    }
    else
    {
        return std::make_unique<NumPyAccessor>(npy_path, shape, data_type, data_layout, output_stream);
    }
}

/** Generates appropriate npy output accessor according to the specified npy_path
 *
 * @note If npy_path is empty will generate a DummyAccessor else will generate a SaveNpyAccessor
 *
 * @param[in] npy_name   Npy filename.
 * @param[in] is_fortran (Optional) If true, save tensor in fortran order.
 *
 * @return An appropriate tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_save_npy_output_accessor(const std::string &npy_name, const bool is_fortran = false)
{
    if(npy_name.empty())
    {
        return std::make_unique<DummyAccessor>(0);
    }
    else
    {
        return std::make_unique<SaveNumPyAccessor>(npy_name, is_fortran);
    }
}

/** Generates print tensor accessor
 *
 * @param[out] output_stream (Optional) Output stream
 *
 * @return A print tensor accessor
 */
inline std::unique_ptr<graph::ITensorAccessor> get_print_output_accessor(std::ostream &output_stream = std::cout)
{
    return std::make_unique<PrintAccessor>(output_stream);
}

/** Permutes a given tensor shape given the input and output data layout
 *
 * @param[in] tensor_shape    Tensor shape to permute
 * @param[in] in_data_layout  Input tensor shape data layout
 * @param[in] out_data_layout Output tensor shape data layout
 *
 * @return Permuted tensor shape
 */
inline TensorShape permute_shape(TensorShape tensor_shape, DataLayout in_data_layout, DataLayout out_data_layout)
{
    if(in_data_layout != out_data_layout)
    {
        arm_compute::PermutationVector perm_vec = (in_data_layout == DataLayout::NCHW) ? arm_compute::PermutationVector(2U, 0U, 1U) : arm_compute::PermutationVector(1U, 2U, 0U);
        arm_compute::permute(tensor_shape, perm_vec);
    }
    return tensor_shape;
}

/** Utility function to return the TargetHint
 *
 * @param[in] target Integer value which expresses the selected target. Must be 0 for Neon or 1 for OpenCL or 2 (OpenCL with Tuner)
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

#endif /* __ARM_COMPUTE_UTILS_GRAPH_UTILS_H__ */
