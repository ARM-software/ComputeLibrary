#include "arm_compute/graph.h"
#ifdef ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/Utils.h"
#endif /* ARM_COMPUTE_CL */
#include "support/ToolchainSupport.h"
#include "utils/CommonGraphOptions.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"

#include "utils/TextLoader.h"

using namespace arm_compute;
using namespace arm_compute::utils;
using namespace arm_compute::graph::frontend;
using namespace arm_compute::graph_utils;


using namespace arm_compute;
using namespace utils;

class NEONInpurExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        UTF8Loader loader;

        if (argc < 2)
        {
            loader.open("./data/test.txt");
            // Create an empty grayscale 640x480 image
            src.allocator()->init(TensorInfo(640, 480, Format::U8));
        }
        else
        {
            loader.open(argv[1]);
            loader.init_text(src,TextFormat::UTF8);
        }

        // Allocate all the images
        src.allocator()->allocate();
        dst.allocator()->allocate();

        // Fill the input image with the content of the PPM image if a filename was provided:
        if (loader.is_open())
        {
            loader.fill_text(src);
        }

        return true;
    }
    void do_run() override
    {
        std::cout << "Testing input" << std::endl;
    }
    void do_teardown() override
    {
        std::cout << "Testing input Ended" << std::endl;
    }

private:
    Text       src{}, dst{};
    std::string output_filename{};
};

/** Main program for convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEONInpurExample>(argc, argv);
}
