#!/bin/bash

set -e

DIRECTORIES="./arm_compute ./src ./examples ./tests ./utils ./framework ./support"

grep -HrnP "/\*\*$" $DIRECTORIES | tee bad_style.log
if (( `cat bad_style.log | wc -l` > 0 ))
then
    echo ""
    echo "ERROR: Doxygen comments should start on the first line: \"/** My comment\""
    exit -1
fi

grep -Hnr --exclude=Doxyfile "@brief" $DIRECTORIES | tee bad_style.log
if (( `cat bad_style.log | wc -l` > 0 ))
then
    echo ""
    echo "ERROR: Doxygen comments shouldn't use '@brief'"
    exit -1
fi

grep -HnRE "\buint " --exclude-dir=cl_kernels $DIRECTORIES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: C/C++ don't define 'uint'. Use 'unsigned int' instead."
    exit -1
fi

grep -HnR "float32_t" $DIRECTORIES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: C/C++ don't define 'float32_t'. Use 'float' instead."
    exit -1
fi

grep -Hnir "arm[_ ]\?cv" $DIRECTORIES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: Reference to arm_cv detected in the files above (Replace with arm_compute)"
    exit -1
fi

grep -Hnir "#.*defined[^(]" $DIRECTORIES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: use parenthesis after #if defined(MY_PREPROCESSOR)"
    exit -1
fi

grep -Hnir "#else$\|#endif$" $DIRECTORIES | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: #else and #endif should be followed by a comment of the guard they refer to (e.g /* ARM_COMPUTE_ENABLE_FP16 */ )"
    exit -1
fi

grep -Hnir "ARM_COMPUTE_ENABLE_FP16" ./tests/validation_new/CL | tee bad_style.log
if [[ $(cat bad_style.log | wc -l) > 0 ]]
then
    echo ""
    echo "ERROR: Found ARM_COMPUTE_ENABLE_FP16 in CL tests though F16 is always supported for OpenCL"
    exit -1
fi

spdx_missing=0
for f in $(find $DIRECTORIES -type f)
do
    if [[ $(grep SPDX $f | wc -l) == 0 ]]
    then
        # List of exceptions:
        case `basename $f` in
            "arm_compute_version.embed");;
            ".clang-format");;
            ".clang-tidy");;
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
