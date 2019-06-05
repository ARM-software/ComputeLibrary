/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

#include <fstream>
#include <sstream>
#include <vector>

using namespace arm_compute;
using namespace utils;

class NeonOpticalFlowExample : public Example
{
public:
    NeonOpticalFlowExample()
        : input_points(100), output_points(100), point_estimates(100)
    {
    }

    bool do_setup(int argc, char **argv) override
    {
        if(argc < 5)
        {
            // Print help
            std::cout << "Usage: ./build/neon_opticalflow [src_1st.ppm] [src_2nd.ppm] [keypoints] [estimates]\n\n";
            const unsigned int img_width  = 64;
            const unsigned int img_height = 64;
            const unsigned int rect_x     = 20;
            const unsigned int rect_y     = 40;
            const unsigned int rect_s     = 8;
            const unsigned int offsetx    = 24;
            const unsigned int offsety    = 3;
            std::cout << "No input_image provided, creating test data:\n";
            std::cout << "\t Image src_1st = (" << img_width << "," << img_height << ")" << std::endl;
            std::cout << "\t Image src_2nd = (" << img_width << "," << img_height << ")" << std::endl;
            init_img(src_1st, img_width, img_height, rect_x, rect_y, rect_s);
            init_img(src_2nd, img_width, img_height, rect_x + offsetx, rect_y + offsety, rect_s);
            const int num_points = 4;
            input_points.resize(num_points);
            point_estimates.resize(num_points);
            const std::array<unsigned int, num_points> tracking_coordsx = { rect_x - 1, rect_x, rect_x + 1, rect_x + 2 };
            const std::array<unsigned int, num_points> tracking_coordsy = { rect_y - 1, rect_y, rect_y + 1, rect_y + 2 };
            const std::array<unsigned int, num_points> estimate_coordsx = { rect_x + offsetx - 1, rect_x + offsetx, rect_x + offsetx + 1, rect_x + offsetx + 2 };
            const std::array<unsigned int, num_points> estimate_coordsy = { rect_y + offsety - 1, rect_y + offsety, rect_y + offsety + 1, rect_y + offsety + 2 };

            for(int k = 0; k < num_points; ++k)
            {
                auto &keypoint           = input_points.at(k);
                keypoint.x               = tracking_coordsx[k];
                keypoint.y               = tracking_coordsy[k];
                keypoint.tracking_status = 1;
            }
            for(int k = 0; k < num_points; ++k)
            {
                auto &keypoint           = point_estimates.at(k);
                keypoint.x               = estimate_coordsx[k];
                keypoint.y               = estimate_coordsy[k];
                keypoint.tracking_status = 1;
            }
        }
        else
        {
            load_ppm(argv[1], src_1st);
            load_ppm(argv[2], src_2nd);
            load_keypoints(argv[3], input_points);
            load_keypoints(argv[4], point_estimates);
        }

        print_points(input_points, "Tracking points : ");
        print_points(point_estimates, "Estimates points : ");

        const unsigned int num_levels = 3;
        // Initialise and allocate pyramids
        PyramidInfo pyramid_info(num_levels, SCALE_PYRAMID_HALF, src_1st.info()->tensor_shape(), src_1st.info()->format());
        pyr_1st.init_auto_padding(pyramid_info);
        pyr_2nd.init_auto_padding(pyramid_info);

        pyrf_1st.configure(&src_1st, &pyr_1st, BorderMode::UNDEFINED, 0);
        pyrf_2nd.configure(&src_2nd, &pyr_2nd, BorderMode::UNDEFINED, 0);

        output_points.resize(input_points.num_values());

        optkf.configure(&pyr_1st, &pyr_2nd,
                        &input_points, &point_estimates, &output_points,
                        Termination::TERM_CRITERIA_BOTH, 0.01f, 15, 5, true, BorderMode::UNDEFINED, 0);

        pyr_1st.allocate();
        pyr_2nd.allocate();

        return true;
    }
    void do_run() override
    {
        //Execute the functions:
        pyrf_1st.run();
        pyrf_2nd.run();
        optkf.run();
    }
    void do_teardown() override
    {
        print_points(output_points, "Output points : ");
    }

private:
    /** Loads the input keypoints from a file into an array
     *
     * @param[in]  fn  Filename containing the keypoints. Each line must have two values X Y.
     * @param[out] img Reference to an unintialised KeyPointArray
     */
    bool load_keypoints(const std::string &fn, KeyPointArray &array)
    {
        assert(!fn.empty());
        std::ifstream f(fn);
        if(f.is_open())
        {
            std::cout << "Reading points from " << fn << std::endl;
            std::vector<KeyPoint> v;
            for(std::string line; std::getline(f, line);)
            {
                std::stringstream ss(line);
                std::string       xcoord;
                std::string       ycoord;
                getline(ss, xcoord, ' ');
                getline(ss, ycoord, ' ');
                KeyPoint kp;
                kp.x               = std::stoi(xcoord);
                kp.y               = std::stoi(ycoord);
                kp.tracking_status = 1;
                v.push_back(kp);
            }
            const int num_points = v.size();
            array.resize(num_points);
            for(int k = 0; k < num_points; ++k)
            {
                auto &keypoint = array.at(k);
                keypoint       = v[k];
            }
            return true;
        }
        else
        {
            std::cout << "Cannot open keypoints file " << fn << std::endl;
            return false;
        }
    }

    /** Creates and Image and fills it with the ppm data from the file
     *
     * @param[in]  fn  PPM filename to be loaded
     * @param[out] img Reference to an unintialised image instance
     */
    bool load_ppm(const std::string &fn, Image &img)
    {
        assert(!fn.empty());
        PPMLoader ppm;
        ppm.open(fn);
        ppm.init_image(img, Format::U8);
        img.allocator()->allocate();
        if(ppm.is_open())
        {
            std::cout << "Reading image " << fn << std::endl;
            ppm.fill_image(img);
            return true;
        }
        else
        {
            std::cout << "Cannot open " << fn << std::endl;
            return false;
        }
    }
    /** Creates and Image and draws a square in the specified coordinares.
     *
     * @param[out] img             Reference to an unintialised image instance
     * @param[in]  img_width       Width of the image to be created
     * @param[in]  img_height      Height of the image to be created
     * @param[in]  square_center_x Coordinate along x-axis to be used as the center for the square
     * @param[in]  square_center_y Coordinate along y-axis to be used as the center for the square
     * @param[in]  square_size     Size in pixels to be used for the square
     */
    void init_img(Image &img, unsigned int img_width, unsigned int img_height,
                  unsigned int square_center_x, unsigned int square_center_y,
                  unsigned int square_size)
    {
        img.allocator()->init(TensorInfo(img_width, img_height, Format::U8));
        img.allocator()->allocate();
        const unsigned int square_half = square_size / 2;
        // assert the square is in the bounds of the image
        assert(square_center_x > square_half && square_center_x + square_half < img_width);
        assert(square_center_y > square_half && square_center_y + square_half < img_height);
        // get ptr to the top left pixel for the squeare
        std::fill(img.buffer(), img.buffer() + img_width * img_height, 0);
        for(unsigned int i = 0; i < square_size; ++i)
        {
            for(unsigned int j = 0; j < square_size; ++j)
            {
                uint8_t *ptr = img.ptr_to_element(Coordinates(square_center_x - square_half + j, square_center_y - square_half + i));
                *ptr         = 0xFF;
            }
        }
    }
    /** Prints an array of keypoints and an optional label
     *
     * @param[in] a   Keypoint array to be printed
     * @param[in] str Label to be printed before the array
     */
    void print_points(const KeyPointArray &a, const std::string &str = "")
    {
        std::cout << str << std::endl;
        for(unsigned int k = 0; k < a.num_values(); ++k)
        {
            auto kp = a.at(k);
            std::cout << "\t "
                      << " (x,y) = (" << kp.x << "," << kp.y << ")";
            std::cout << " strength = " << kp.strength << " "
                      << " scale = " << kp.scale << " orientation " << kp.orientation << " status " << kp.tracking_status << " err = " << kp.error << std::endl;
        }
    }

    Pyramid               pyr_1st{};
    Pyramid               pyr_2nd{};
    NEGaussianPyramidHalf pyrf_1st{};
    NEGaussianPyramidHalf pyrf_2nd{};
    NEOpticalFlow         optkf{};
    Image                 src_1st{}, src_2nd{};
    KeyPointArray         input_points;
    KeyPointArray         output_points;
    KeyPointArray         point_estimates;
};

/** Main program for optical flow test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NeonOpticalFlowExample>(argc, argv);
}
