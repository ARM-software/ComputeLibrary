# Copyright (c) 2019 ARM Limited.
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
ALL_STRATEGY_OPTIONS=("reshaped_rhs_only" "reshaped")

# Names of example binary for each strategy
EXAMPLE_BIN_RESHAPED_RHS_ONLY="benchmark_cl_gemm_reshaped_rhs_only"
EXAMPLE_BIN_RESHAPED="benchmark_cl_gemm_reshaped"

# Default output directory
DEFAULT_OUT_DIR="out"

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
  Gemm shape file is a headerless csv file with fields separated by commas and commas only (there cannot be whitespaces
  around each field).
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
  Gemm config file is a headerless csv file with fields separated by commas and commas only (there cannot be whitespaces
  around each field).
  A gemm config is a list of 4 positive integers <m0, n0, k0, h0> and 2 boolean values interleave_rhs and transpose_rhs, with:
  m0 - Number of rows processed by the matrix multiplication
  n0 - Number of columns processed by the matrix multiplication
  k0 - Number of partial accumulations performed by the matrix multiplication
  h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row
  interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)
  transpose_rhs - Transpose rhs matrix (1) / Do not transpose rhs matrix (0)

  Only the following configurations of M0, N0 and K0 are currently supported:
  M0 = 1, 2, 3, 4, 5, 6, 7, 8
  N0 = 2, 3, 4, 8, 16
  K0 = 2, 3, 4, 8, 16
  H0 >= 1

  An example gemm config file looks like:
  4,4,4,1,1,1
  4,4,4,3,1,0
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
  Gemm config file is a headerless csv file with fields separated by commas and commas only (there cannot be whitespaces
  around each field).
  A gemm config is a list of 5 positive integers <m0, n0, k0, v0, h0> and 3 boolean values interleave_lhs, interleave_rhs and transpose_rhs, with:
  m0 - Number of rows processed by the matrix multiplication
  n0 - Number of columns processed by the matrix multiplication
  k0 - Number of partial accumulations performed by the matrix multiplication
  v0 - Number of vertical blocks of size (m0xk0) stored on the same output row
  h0 - Number of horizontal blocks of size (k0xn0) stored on the same output row
  interleave_lhs - Interleave lhs matrix (1) / Do not interleave lhs matrix (0)
  interleave_rhs - Interleave rhs matrix (1) / Do not interleave rhs matrix (0)
  transpose_rhs - Transpose rhs matrix but not lhs matrix (1) / Do not transpose rhs matrix but do transpose lhs matrix (0)

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
  4,4,4,1,3,1,1,1
  4,4,4,3,3,1,1,0
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

Usage: ${CMD} [-h] -s <strategy> -e <example_binary_dir> -g <gemm_shape_file> -c <gemm_config_file> [-o <out_dir>]

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

        -o <out_dir>
        Path to output directory that holds output json files
        Default: ${DEFAULT_OUT_DIR}

EOF
# Print help messages about gemm shapes and various gemm configs
$HELP && help_gemm_shape_file
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
# Run all tunable configurations and all input configurations
# Globals:
#   OUT_DIR
#   EXAMPLE_BIN_DIR
#   NUM_ITERATION
#   GEMM_CONFIGS_FILE
#   GEMM_SHAPES_FILE
# Arguments:
#   example_bin   Name of the example binary to run
# Returns:
#   None
#######################################
function run() {
  local example_bin=$1
  echo "Running all configs for ${example_bin}" 1>&2
  local example_args
  local test_id=1
  while read gemm_shape
  do
    while read gemm_config
    do
      example_args="${gemm_shape},${gemm_config}"
      ${EXAMPLE_BIN_DIR}/${example_bin} --example_args=${example_args} --iterations=${NUM_ITERATION} --json-file=${OUT_DIR}/${test_id} --instruments=OPENCL_TIMER_MS
      (( test_id++ ))
    done < "${GEMM_CONFIGS_FILE}"
  done < "${GEMM_SHAPES_FILE}"
  echo "Finished running all configs for ${example_bin}" 1>&2
  echo "All results saved to ${OUT_DIR}" 1>&2
}

# Functions }}}

# Main: Main script {{{
# Path to directory containing all benchmark examples binaries
EXAMPLE_BIN_DIR=""
# Path to gemm shapes file
GEMM_SHAPES_FILE=""
# Path to gemm configs file
GEMM_CONFIGS_FILE=""
STRATEGY_OPTION=""
# Path to output directory
OUT_DIR=${DEFAULT_OUT_DIR}
# Toggle help
HELP=false

# Obtain options
while getopts "hs:e:g:c:o:" opt; do
  case "$opt" in
    h) HELP=true ;;
    s) STRATEGY_OPTION=$(to_lower "${OPTARG}");;
    e) EXAMPLE_BIN_DIR="${OPTARG}";;
    g) GEMM_SHAPES_FILE="${OPTARG}";;
    c) GEMM_CONFIGS_FILE="${OPTARG}";;
    o) OUT_DIR="${OPTARG}";;
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

# Make sure existing benchmark outputs are not overwritten
[ ! -d "${OUT_DIR}" ] ||
  error_msg "Output directory ${OUT_DIR} already exists!"

# Make output directory
mkdir ${OUT_DIR}

# Run selected strategy with all configurations
[ "${STRATEGY_OPTION}" == "reshaped_rhs_only" ] && run $EXAMPLE_BIN_RESHAPED_RHS_ONLY
[ "${STRATEGY_OPTION}" == "reshaped" ] && run $EXAMPLE_BIN_RESHAPED
# Main: Main script }}}
