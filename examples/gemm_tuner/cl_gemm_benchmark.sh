# Copyright (c) 2019-2021 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/bin/sh

# Global: Global variables and global settings {{{
# Treat unset variables as an error when substituting
set -u

CMD=$( basename $0 )

# All supported strategy options
ALL_STRATEGY_OPTIONS=("native" "reshaped_rhs_only" "reshaped")

# All supported data type options
ALL_DATA_TYPE_OPTIONS=("f32" "f16" "qasymm8")

# Names of example binary for each strategy
EXAMPLE_BIN_NATIVE="benchmark_cl_gemm_native"
EXAMPLE_BIN_RESHAPED_RHS_ONLY="benchmark_cl_gemm_reshaped_rhs_only"
EXAMPLE_BIN_RESHAPED="benchmark_cl_gemm_reshaped"
EXAMPLE_BIN_RESHAPED_RHS_ONLY_LOWP="benchmark_cl_gemmlowp_reshaped_rhs_only_fused_output_stage_fixedpoint"
EXAMPLE_BIN_RESHAPED_LOWP="benchmark_cl_gemmlowp_reshaped"

# Default data type
DEFAULT_DATA_TYPE="f32"

# Default output directory
DEFAULT_OUT_DIR="out"

# Default ID of the first experiment
DEFAULT_ID_EXPERIMENT_START=0

# Default total number of experiments
DEFAULT_NUM_EXPERIMENTS="all"

# Default output file extension
DEFAULT_OUT_EXTENSION="mlgo_benchmark"

# Default OpenCL tuner mode
DEFAULT_TUNER_MODE="rapid"

# Number of iterations for each benchmark run
NUM_ITERATION=5
# Global }}}

# Functions {{{
#######################################
# Print gemm shape file help message
# Globals:
#   None
# Arguments:
#   None
# Returns:
#   None
#######################################
function help_gemm_shape_file() {
  cat >&2 << EOF
Gemm shape file:
  Gemm shape file is a csv file with fields separated by commas. The optional header and comments are ignored by the parser.

  A gemm shape is a list of 4 positive integers <M, N, K, B> describing the shapes of the two matrices (LHS and RHS)
  with:
  M - Number of lhs matrix rows
  N - Number of rhs matrix columns
  K - Number of lhs matrix columns/rhs matrix rows
  B - Batch size

  An example gemm shape file looks like:
  100,100,30,1
  100,100,30,3
  ...

EOF
}

#######################################
# Print gemm config file for native help message
# Globals:
#   None
# Arguments:
#   None
# Returns:
#   None
#######################################
function help_gemm_config_file_native() {
  cat >&2 << EOF
Gemm config file (Strategy native):
  Gemm config file is a csv file with fields separated by commas. The optional header and comments are ignored by the parser.

  A gemm config is a list of 3 positive integers <m0, n0, k0>, with:
  m0 - Number of rows processed by the matrix multiplication
  n0 - Number of columns processed by the matrix multiplication
  k0 - Number of partial accumulations performed by the matrix multiplication

  Only the following configurations of M0, N0 and K0 are currently supported:
  M0 = 1, 2, 3, 4, 5, 6, 7, 8
  N0 = 2, 3, 4, 8, 16
  K0 = 2, 3, 4, 8, 16

  An example gemm config file looks like:
  1,4,4
  2,3,8
  ...

EOF
}

#######################################
# Print gemm config file for reshaped_rhs_only help message
# Globals:
#   None
# Arguments:
#   None
# Returns:
#   None
#######################################
function help_gemm_config_file_reshaped_rhs_only() {
  cat >&2 << EOF
Gemm config file (Strategy reshaped_rhs_only):
  Gemm config file is a csv file with fields separated by commas. The optional header and comments are ignored by the parser.

  A gemm config is a list of 4 positive integers <m0, n0, k0, h0> and 3 boolean values:
  m0 - Number of rows processed by the matrix multiplication
  n0 - Number of columns processed by the matrix multiplication
  k0 - Number of partial accumulations performed by the matrix multiplication
  h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row
  interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)
  transpose_rhs - Transpose rhs matrix (1) / Do not transpose rhs matrix (0)
  export_to_cl_image_rhs - (Not supported for quantized types) Export rhs matrix to cl_image (1) / Do not export rhs matrix to cl_image (0). Can only be true
                           with certain combinations of the GEMMParams and other configs. Please refer to CLGEMMReshapeRHSMatrixKernel
                           for more details

  Only the following configurations of M0, N0 and K0 are currently supported:
  M0 = 1, 2, 3, 4, 5, 6, 7, 8
  N0 = 2, 3, 4, 8, 16
  K0 = 2, 3, 4, 8, 16
  H0 >= 1

  An example gemm config file looks like:
  4,4,4,1,1,1,0
  4,4,4,3,1,0,1
  ...

EOF
}

#######################################
# Print gemm config file for reshaped help message
# Globals:
#   None
# Arguments:
#   None
# Returns:
#   None
#######################################
function help_gemm_config_file_reshaped() {
  cat >&2 << EOF
Gemm config file (Strategy reshaped):
  Gemm config file is a csv file with fields separated by commas. The header and comments are ignored by the parser.

  A gemm config is a list of 5 positive integers <m0, n0, k0, v0, h0> and 4 boolean values:
  m0 - Number of rows processed by the matrix multiplication
  n0 - Number of columns processed by the matrix multiplication
  k0 - Number of partial accumulations performed by the matrix multiplication
  v0 - Number of vertical blocks of size (m0xk0) stored on the same output row
  h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row
  interleave_lhs - Interleave lhs matrix (1) / Do not interleave lhs matrix (0)
  interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)
  transpose_rhs - Transpose rhs matrix but not lhs matrix (1) / Do not transpose rhs matrix but do transpose lhs matrix (0)
  export_to_cl_image_rhs - (Not supported for quantized types) Export rhs matrix to cl_image (1) / Do not export rhs matrix to cl_image (0). Can only be true
                           with certain combinations of the GEMMParams and other configs. Please refer to CLGEMMReshapeRHSMatrixKernel
                           for more details

  If rhs matrix is transposed only the following configurations are currently supported:
  M0 = 2, 3, 4, 5, 6, 7, 8
  N0 = 2, 3, 4, 8, 16
  K0 = 2, 3, 4, 8, 16
  V0 >= 1
  H0 >= 1

  If lhs matrix is transposed only the following configurations are currently supported:
  M0 = 2, 3, 4, 8
  N0 = 2, 3, 4, 8, 16
  K0 = 2, 3, 4, 8, 16
  V0 >= 1
  H0 >= 1

  An example gemm config file looks like:
  4,4,4,1,3,1,1,1,0
  4,4,4,3,3,1,1,0,1
  ...

EOF
}

#######################################
# Print usage of this program and exit with Error
# Globals:
#   Assumes all globals are required
# Arguments:
#   None
# Returns:
#   Error(1)
#######################################
function usage() {
  cat >&2 << EOF
Run gemm examples of a selected strategy, over provided tunable configurationsa and gemm shapes.
Save the benchmark results to json files in an output directory.

Usage: ${CMD} [-h] -s <strategy> -e <example_binary_dir> -g <gemm_shape_file> -c <gemm_config_file> [-o <out_dir>] [-d <data_type>] [-i <id_experiment_start>] [-n <num_experiments>] [-t <output_extension>]

Options:
        -h
        Print help messages. If a strategy is specified with -s <strategy>, then only display messages relevant to that
        strategy. Otherwise if no strategy is specified, display messages for all available strategies.

        -s <strategy>
        Strategy option.
        Options: ${ALL_STRATEGY_OPTIONS[@]}.

        -e <example_binary_dir>
        Path to directory that holds all example binaries

        -g <gemm_shape_file>
        Path to gemm shape csv file

        -c <gemm_config_file>
        Path to gemm config csv file

        -d <data_type>
        Data type option with which to run benchmark examples
        Default: ${DEFAULT_DATA_TYPE}
        Supported options:
        Strategy            :    Data Types
        Native              :    f32
        Reshaped            :    f32, f16, qasymm8
        Reshaped RHS Only   :    f32, f16, qasymm8

        -o <out_dir>
        Path to output directory that holds output json files
        Default: ${DEFAULT_OUT_DIR}

        -i <id_experiment_start>
        ID of the first experiment.
        Default: ${DEFAULT_ID_EXPERIMENT_START}

        -n <num_experiments>
        Total number of experiments to execute in this session. [1-all]
        Default: ${DEFAULT_NUM_EXPERIMENTS}

        -t <output_extension>
        Output file extension.
        Default: ${DEFAULT_OUT_EXTENSION}

        -m <tuner_mode>
        OpenCL tuner mode.
        Default: ${DEFAULT_TUNER_MODE}

EOF
# Print help messages about gemm shapes and various gemm configs
$HELP && help_gemm_shape_file
$HELP && ( [ "${STRATEGY_OPTION}" == "" ] || [ "${STRATEGY_OPTION}" == "native" ] ) && help_gemm_config_file_native
$HELP && ( [ "${STRATEGY_OPTION}" == "" ] || [ "${STRATEGY_OPTION}" == "reshaped_rhs_only" ] ) && help_gemm_config_file_reshaped_rhs_only
$HELP && ( [ "${STRATEGY_OPTION}" == "" ] || [ "${STRATEGY_OPTION}" == "reshaped" ] ) && help_gemm_config_file_reshaped
exit 1
}

#######################################
# Print error message and exit with Error.
# Globals:
#   None
# Arguments:
#   $1 - Error message
# Returns:
#   None
#######################################
function error_msg() {
  echo "Error: $1" 1>&2
  exit 1
}

#######################################
# Convert string to lower-case
# Globals:
#   None
# Arguments:
#   target  - String
# Returns:
#   (stdout) - String in lowercase
#######################################
function to_lower() {
  local target=$1
  echo "$target" | tr '[:upper:]' '[:lower:]'
}

#######################################
# Test if the argument is an integer
# Globals:
#   None
# Arguments:
#   in   - Input
# Returns:
#   true/false
#######################################
function is_integer() {
  local in=$1
  [ "$in" -eq "$in" ] 2> /dev/null
}

#######################################
# Test if a string is in an array of strings
# Globals:
#   None
# Arguments:
#   target  - String to test
#   array   - Array of strings to search
# Returns:
#   true/false
#######################################
function arr_contains() {
  local target=$1
  shift
  local array
  array=("$@")
  for s in "${array[@]}"
  do
    [ "$s" == "${target}" ] && return
  done
  false
}

#######################################
# Run a single example with all tunable gemm configurations on all gemm parameters
# Globals:
#   OUT_DIR
#   OUT_EXTENSION
#   TUNER_MODE
#   EXAMPLE_BIN_DIR
#   NUM_ITERATION
#   GEMM_CONFIGS_FILE
#   GEMM_SHAPES_FILE
#   STRATEGY_OPTION
#   DATA_TYPE
#   OUT_DIR
#   ID_EXPERIMENT_START
#   NUM_EXPERIMENTS

# Arguments:
#   example_bin   Name of the example binary to run
# Returns:
#   None
#######################################
function run() {
  local example_bin=$1
  echo "Running experiments for ${example_bin}" 1>&2
  local example_args
  local json_filename
  local expr_count=0
  # Total number of experiments available
  local num_experiments_total
  # Total number of experiment runs scheduled for this session
  local num_experiments_session
  local id_experiment_start
  local id_experiment_end
  local array_shapes
  local array_configs
  local array_shapes_len
  local array_configs_len
  local array_shapes_idx
  local array_configs_idx
  local match_expression_shape="^([^,]*,){3}[^,]*$"
  local match_expression_config="^(\s*[0-9]+\s*,)+\s*[0-9]\s*$"
  local shapes_list_cmd="grep -E "$match_expression_shape" "${GEMM_SHAPES_FILE}""
  local configs_list_cmd="grep -E "$match_expression_config" "${GEMM_CONFIGS_FILE}""

  # Create array from CSV file
  array_shapes=($( $shapes_list_cmd ))
  array_configs=($( $configs_list_cmd ))

  # Get array length
  array_shapes_len=${#array_shapes[@]}
  array_configs_len=${#array_configs[@]}

  # Get the total number of experiments available
  (( num_experiments_total=${array_shapes_len} * ${array_configs_len} ))

  # Get the number of experiments to execute in this session
  if [ ${NUM_EXPERIMENTS} == ${DEFAULT_NUM_EXPERIMENTS} ]
  then
    (( num_experiments_session=${array_shapes_len} * ${array_configs_len} ))
  else
    num_experiments_session=$NUM_EXPERIMENTS
  fi

  # Id experiment start
  id_experiment_start=${ID_EXPERIMENT_START}

  # Id experiment end
  (( id_experiment_end=(${num_experiments_session} + ${id_experiment_start} - 1) ))

  # Check if the id experiment end is grater than or equal to the total number of experiments available.
  # If the condition is satisfied, clamp the id experiment end
  if [ "$id_experiment_end" -ge "$num_experiments_total" ]
  then
    echo "Clamping idx experiment end" 1>&2
    (( id_experiment_end=${num_experiments_total} - 1 ))
    (( num_experiments_session=${id_experiment_start} + ${id_experiment_end} + 1 ))
  fi

  # Time elapsed since the beginning in seconds
  local time_elapsed_s
  # Time estimated to finish in seconds
  local time_est_s
  echo "Running a total number of ${num_experiments_session} experiments" 1>&2
  echo "Experiment idx start/end [${id_experiment_start}, ${id_experiment_end}]" 1>&2

  # Run experiments
  for i in $(seq $id_experiment_start $id_experiment_end);
  do
    (( array_shapes_idx=${i} / ${array_configs_len} ))
    (( array_configs_idx=${i} % ${array_configs_len} ))

    gemm_shape=${array_shapes[$array_shapes_idx]}
    gemm_config=${array_configs[$array_configs_idx]}

    echo "Running shape[$array_shapes_idx]=$gemm_shape with config[$array_configs_idx]=$gemm_config" 1>&2

    example_args="${gemm_shape},${gemm_config},--type=${DATA_TYPE},--tuner-mode=${TUNER_MODE}"
    json_filename="${STRATEGY_OPTION}_${gemm_shape}_${gemm_config}_${DATA_TYPE}"
    # Replace "," with "_"
    json_filename=${json_filename//,/_}

    # Run experiment
    ${EXAMPLE_BIN_DIR}/${example_bin} --example_args=${example_args} --iterations=${NUM_ITERATION} --json-file=${OUT_DIR}/${json_filename}.${OUT_EXTENSION} --instruments=OPENCL_TIMER_MS
    # Print progress
    (( expr_count++ ))
    print_progress ${expr_count} ${num_experiments_session}
    # Print time statistics
    time_elapsed_s=$SECONDS
    echo "Time elapsed since beginning: $(( $time_elapsed_s / 60 ))m $(( $time_elapsed_s % 60 ))s" 1>&2
    (( time_est_s=(${num_experiments_session} - ${expr_count}) * ${time_elapsed_s} / ${expr_count} ))
    echo "Time estimated to finish: $(( $time_est_s / 60 ))m $(( $time_est_s % 60 ))s" 1>&2
    echo "Done." 1>&2
  done

  echo "Finished running all configs for ${example_bin}" 1>&2
  echo "All results saved to ${OUT_DIR}" 1>&2
}

#######################################
# Print the progress of the current session
# Globals:
#   None
# Arguments:
#   current   Current number of items
#   total     Total number of items
# Returns:
#   None
#######################################
function print_progress() {
  local current
  local total
  current=$1
  total=$2
  # Width of progress bar
  local width
  width=20
  (( current_width= $width * current / total ))
  echo -n -e "Progress [" 1>&2
  for i in $(seq 1 ${width}); do
    if [[ $i -le ${current_width} ]]; then
      echo -n "#" 1>&2
    else
      echo -n " " 1>&2
    fi
  done
  echo  "] $current / $total Experiments" 1>&2
}

# Functions }}}

# Main: Main script {{{
# Path to directory containing all benchmark examples binaries
EXAMPLE_BIN_DIR=""
# Path to gemm shapes file
GEMM_SHAPES_FILE=""
# Path to gemm configs file
GEMM_CONFIGS_FILE=""
# Strategy option
STRATEGY_OPTION=""
# Data type to use
DATA_TYPE=${DEFAULT_DATA_TYPE}
# Path to output directory
OUT_DIR=${DEFAULT_OUT_DIR}
# ID of the first experiment
ID_EXPERIMENT_START=${DEFAULT_ID_EXPERIMENT_START}
# Total number of experiments to execute in this session
NUM_EXPERIMENTS=${DEFAULT_NUM_EXPERIMENTS}
# Output benchmark result file extension
OUT_EXTENSION=${DEFAULT_OUT_EXTENSION}
# OpenCL tuner mode
TUNER_MODE=${DEFAULT_TUNER_MODE}
# Toggle help
HELP=false

# Obtain options
while getopts "hs:e:g:c:d:o:i:n:t:m:" opt; do
  case "$opt" in
    h) HELP=true ;;
    s) STRATEGY_OPTION=$(to_lower "${OPTARG}");;
    e) EXAMPLE_BIN_DIR="${OPTARG}";;
    g) GEMM_SHAPES_FILE="${OPTARG}";;
    c) GEMM_CONFIGS_FILE="${OPTARG}";;
    d) DATA_TYPE=$(to_lower "${OPTARG}");;
    o) OUT_DIR="${OPTARG}";;
    i) ID_EXPERIMENT_START="${OPTARG}";;
    n) NUM_EXPERIMENTS="${OPTARG}";;
    t) OUT_EXTENSION="${OPTARG}";;
    m) TUNER_MODE="${OPTARG}";;
  esac
done
shift $((OPTIND - 1))

# Lazily print usage (after arguments have been parsed)
$HELP &&
  usage

# Parse and validate options
# Verify all compulsory arguments are passed in
( [ ! -z "${STRATEGY_OPTION}" ] && [ ! -z "${EXAMPLE_BIN_DIR}" ] && [ ! -z "${GEMM_SHAPES_FILE}" ] && [ ! -z "${GEMM_CONFIGS_FILE}" ] ) ||
  usage

# Verify example binaries directory exists
[ -d "${EXAMPLE_BIN_DIR}" ] ||
  error_msg "${EXAMPLE_BIN_DIR} does not exist."

# Verify all benchmark example binaries exist
[ -f "${EXAMPLE_BIN_DIR}/${EXAMPLE_BIN_RESHAPED_RHS_ONLY}" ] ||
  error_msg "Cannot find ${EXAMPLE_BIN_RESHAPED_RHS_ONLY} at ${EXAMPLE_BIN_DIR}"

# Verify Gemm shapes file exists
[ -f "${GEMM_SHAPES_FILE}" ] ||
  error_msg "Cannot find gemm shapes file ${GEMM_SHAPES_FILE}"

# Verify Gemm configs file exists
[ -f "${GEMM_CONFIGS_FILE}" ] ||
  error_msg "Cannot find gemm configs file ${GEMM_CONFIGS_FILE}"

# Verify strategy option is valid
arr_contains "${STRATEGY_OPTION}" "${ALL_STRATEGY_OPTIONS[@]}" ||
  error_msg "Does not support strategy ${STRATEGY_OPTION}"

# Verify data type option is valid
arr_contains "${DATA_TYPE}" "${ALL_DATA_TYPE_OPTIONS[@]}" ||
  error_msg "Does not support data type ${DATA_TYPE}"

# Make sure existing benchmark outputs are not overwritten
[ ! -d "${OUT_DIR}" ] ||
  error_msg "Output directory ${OUT_DIR} already exists!"

# Make output directory
echo "Making output directory ${OUT_DIR}" 1>&2
mkdir -p ${OUT_DIR} || error_msg "Failed to make output directory ${OUT_DIR}"

# Run selected strategy with all configurations
# Restart the built-in timer
SECONDS=0
if [ "$DATA_TYPE" == "qasymm8" ]; then
  [ "${STRATEGY_OPTION}" == "reshaped_rhs_only" ] && run $EXAMPLE_BIN_RESHAPED_RHS_ONLY_LOWP
  [ "${STRATEGY_OPTION}" == "reshaped" ] && run $EXAMPLE_BIN_RESHAPED_LOWP
else
  [ "${STRATEGY_OPTION}" == "native" ] && run $EXAMPLE_BIN_NATIVE
  [ "${STRATEGY_OPTION}" == "reshaped_rhs_only" ] && run $EXAMPLE_BIN_RESHAPED_RHS_ONLY
  [ "${STRATEGY_OPTION}" == "reshaped" ] && run $EXAMPLE_BIN_RESHAPED
fi
# Main: Main script }}}
