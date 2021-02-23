#!/bin/bash

set -e

ALL_DIRECTORIES="./arm_compute ./src ./examples ./tests ./utils ./support"

#If no arguments were passed: default to check all the folders:
if [ ! -n "$1" ]
then
    FILES=$ALL_DIRECTORIES
else
    #else only check the files that were passed on the command line:
    FILES=$@
fi

grep -HrnP --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv "/\*\*$" $FILES | tee bad_style.log
if (( `cat bad_style.log | wc -l` > 0 ))
then
    echo ""
    echo "ERROR: Doxygen comments should start on the first line: \"/** My comment\""
    exit -1
fi

grep -Hnr --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv --exclude=Doxyfile "@brief" $FILES | tee bad_style.log
if (( `cat bad_style.log | wc -l` > 0 ))
then
    echo ""
    echo "ERROR: Doxygen comments shouldn't use '@brief'"
    exit -1
fi

grep -HnRE --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv "\buint " --exclude-dir=cl_kernels --exclude-dir=cs_shaders $FILES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: C/C++ don't define 'uint'. Use 'unsigned int' instead."
    exit -1
fi

grep -HnR --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv "/^float32_t/" $FILES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: C/C++ don't define 'float32_t'. Use 'float' instead."
    exit -1
fi

grep -Hnir --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv "arm[_ ]\?cv" $FILES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: Reference to arm_cv detected in the files above (Replace with arm_compute)"
    exit -1
fi

grep -Hnir --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv "#.*if.*defined[^(]" $FILES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: use parenthesis after #if defined(MY_PREPROCESSOR)"
    exit -1
fi

grep -Hnir --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv "#else$\|#endif$" $FILES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: #else and #endif should be followed by a comment of the guard they refer to (e.g /* ARM_COMPUTE_AARCH64_V8_2 */ )"
    exit -1
fi

grep -Hnir --exclude-dir=assembly --exclude-dir=convolution --exclude-dir=arm_gemm --exclude-dir=arm_conv "ARM_COMPUTE_AARCH64_V8_2" ./tests/validation/CL | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: Found ARM_COMPUTE_AARCH64_V8_2 in CL tests though armv8.2 features (FP16) are always supported for OpenCL"
    exit -1
fi

spdx_missing=0
for f in $(find $FILES -type f)
do
    if [[ $(grep SPDX $f | wc -l) == 0 ]]
    then
        # List of exceptions:
        case `basename $f` in
            "arm_compute_version.embed");;
            ".clang-format");;
            ".clang-tidy");;
            "README.md");;
            #It's an error for other files to not contain the MIT header:
            *)
                spdx_missing=1
                echo $f;
                ;;
        esac
    fi;
done

if [[ $spdx_missing > 0 ]]
then
    echo ""
    echo "ERROR: MIT Copyright header missing from the file(s) above."
    exit -1
fi
