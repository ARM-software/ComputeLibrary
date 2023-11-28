#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Arm Limited.
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

"""
Updates the Doxygen documentation pages with a table of operators supported by Compute Library.

The script builds up a table in XML format internally containing the different operators and their respective supported
compute backends, data types and layouts, and the equivalent operator in the Android Neural Networks API. The list of
operators is pulled from the OperatorList.h header file and further implementation details are provided in the function
headers for the backend-specific operator e.g., NEStridedSlice.h.

Usage:
    python update_supported_ops.py
"""

import argparse
import logging
import re
from enum import Enum
from pathlib import Path


class States(Enum):
    INIT = 0
    DESCRIPTION = 1
    DESCRIPTION_END = 2
    IN_CLASS = 3
    DATA_TYPE_START = 4
    DATA_TYPE_END = 5
    NN_OPERATOR = 6
    NN_OPERATOR_END = 7
    SKIP_OPERATOR = 8
    DATA_LAYOUT_START = 9
    DATA_LAYOUT_END = 10


class OperatorsTable:
    def __init__(self):
        self.project_dir = Path(__file__).parents[1]  # ComputeLibrary directory
        self.xml = ""

    def generate_operator_list(self):
        operator_list_head_file = self.project_dir / "arm_compute" / "runtime" / "OperatorList.h"
        neon_file_name_prefix = str(self.project_dir / "arm_compute" / "runtime" / "NEON" / "functions" / "NE")
        cl_file_name_prefix = str(self.project_dir / "arm_compute" / "runtime" / "CL" / "functions" / "CL")

        logging.debug(operator_list_head_file)

        f = open(operator_list_head_file, 'r')
        # Iterates over the lines of the file
        state = States.INIT
        operator_desc = ""
        nn_op_list = []
        for line in f:
            # /** ActivationLayer
            # *
            # * Description:
            # * Function to simulate an activation layer with the specified activation function.
            # *
            # * Equivalent Android NNAPI Op:
            # * ANEURALNETWORKS_ELU
            # * ANEURALNETWORKS_HARD_SWISH
            # * ANEURALNETWORKS_LOGISTIC
            # * ANEURALNETWORKS_RELU
            # * ANEURALNETWORKS_RELU1
            # * ANEURALNETWORKS_RELU6
            # * ANEURALNETWORKS_TANH
            # *
            # */
            # Check for "/**" of the start of the operator
            r = re.search('^\s*/\*\*(.*)', line)
            if r and state == States.INIT:
                # Skip below ones
                if re.search('.*\(not ported\)', line):
                    state = States.SKIP_OPERATOR
                    continue
                if re.search('.*\(only CL\)', line):
                    state = States.SKIP_OPERATOR
                    continue
                if re.search('.*\(no CL\)', line):
                    state = States.SKIP_OPERATOR
                    continue
                if re.search('.*\(skip\)', line):
                    state = States.SKIP_OPERATOR
                    continue
            # Check" */"
            r = re.match('\s*\*/\s*$', line)
            if r and state == States.SKIP_OPERATOR:
                state = States.INIT
                continue
            # Check " *"
            r = re.match('\s*\*\s*$', line)
            if r and state == States.SKIP_OPERATOR:
                continue
            # Check non " *" lines
            r = re.search('^\s*\*(.*)', line)
            if r and state == States.SKIP_OPERATOR:
                continue

            # Check for "/**" of the start of the operator
            r = re.search('^\s*/\*\*(.*)', line)
            if r and state == States.INIT:
                tmp = r.groups()[0]
                class_name = tmp.strip()
                logging.debug(class_name)
                continue

            # Check whether "Description: " exists
            r = re.search('\s*\*\s*Description:\s*', line)
            if r and state == States.INIT:
                state = States.DESCRIPTION
                continue
            # Treat description ends with a blank line only with " *"
            r = re.match('\s*\*\s*$', line)
            if r and state == States.DESCRIPTION:
                logging.debug(operator_desc)
                state = States.DESCRIPTION_END
                continue
            # Find continuing class description in the following lines
            r = re.search('^\s*\*(.*)', line)
            if r and state == States.DESCRIPTION:
                tmp = r.groups()[0]
                operator_desc = operator_desc + ' ' + tmp.strip()
                continue

            # Check whether "Equivalent AndroidNN Op: " exists
            r = re.search('\s*\*\s*Equivalent Android NNAPI Op:\s*', line)
            if r and state == States.DESCRIPTION_END:
                state = States.NN_OPERATOR
                continue
            # Treat AndroidNN Op ends with a blank line only with " *"
            r = re.match('\s*\*\s*$', line)
            if r and state == States.NN_OPERATOR:
                logging.debug(nn_op_list)
                state = States.NN_OPERATOR_END
                # Check NE#class_name
                neon_file_name = neon_file_name_prefix + class_name + ".h"
                logging.debug(neon_file_name)
                # Check CL#class_name
                cl_file_name = cl_file_name_prefix + class_name + ".h"
                logging.debug(cl_file_name)
                # Check whether CL/Neon file exists
                if Path(neon_file_name).is_file() and Path(cl_file_name).is_file():
                    if neon_file_name.find("NEElementwiseOperations.h") != -1:
                        logging.debug(neon_file_name)
                        self.generate_operator_common_info(class_name, operator_desc, nn_op_list, "13")
                    elif neon_file_name.find("NEElementwiseUnaryLayer.h") != -1:
                        logging.debug(neon_file_name)
                        self.generate_operator_common_info(class_name, operator_desc, nn_op_list, "8")
                    else:
                        self.generate_operator_common_info(class_name, operator_desc, nn_op_list, "2")
                    self.generate_operator_info(neon_file_name)
                    self.generate_operator_cl_begin()
                    self.generate_operator_info(cl_file_name)
                else:
                    if neon_file_name.find("NELogical.h") != -1:
                        logging.debug(neon_file_name)
                        self.generate_operator_common_info(class_name, operator_desc, nn_op_list, "3")
                    else:
                        self.generate_operator_common_info(class_name, operator_desc, nn_op_list, "1")
                    if Path(neon_file_name).is_file():
                        self.generate_operator_info(neon_file_name)
                    if Path(cl_file_name).is_file():
                        self.generate_operator_info(cl_file_name)
                continue

            # Find continuing AndroidNN Op in the following lines
            r = re.search('^\s*\*(.*)', line)
            if r and state == States.NN_OPERATOR:
                tmp = r.groups()[0]
                nn_op = tmp.strip()
                nn_op_list.append(nn_op)
                continue

            # Treat operator ends with a blank line only with " */"
            r = re.match('\s*\*/\s*$', line)
            if r and state == States.NN_OPERATOR_END:
                operator_desc = ""
                nn_op_list = []
                state = States.INIT
                continue
        f.close()

    def generate_operator_info(self, file_name):
        logging.debug(file_name)
        f = open(file_name, 'r')
        # iterates over the lines of the file
        state = States.INIT
        data_type_list = []
        data_layout_list = []
        io_list = []
        class_no = 0
        for line in f:
            # Locate class definition by "class...: public IFunction",
            # There are also exceptions, which will need to support in later version
            r = re.match("\s*class\s+(\S+)\s*:\s*(public)*", line)
            if r and state == States.INIT:
                class_name = r.groups()[0]
                logging.debug("class name is %s" % (class_name))
                state = States.IN_CLASS
                continue

            r = re.match("\s*\}\;", line)
            if r and state == States.IN_CLASS:
                state = States.INIT
                continue

            # * Valid data layouts:
            # * - All
            r = re.search('\s*\*\s*Valid data layouts:', line)
            if r and state == States.IN_CLASS:
                state = States.DATA_LAYOUT_START
                continue
            # Treat data configuration ends with a blank line only with " *"
            r = re.match('\s*\*\s*$', line)
            if r and state == States.DATA_LAYOUT_START:
                state = States.DATA_LAYOUT_END
                continue
            # Data layout continues
            r = re.search('\s*\*\s*\-\s*(.*)', line)
            if r and state == States.DATA_LAYOUT_START:
                tmp = r.groups()[0]
                tmp = tmp.strip()
                logging.debug(tmp)
                data_layout_list.append(tmp)

            # * Valid data type configurations:
            # * |src0           |dst            |
            # * |:--------------|:--------------|
            # * |QASYMM8        |QASYMM8        |
            # * |QASYMM8_SIGNED |QASYMM8_SIGNED |
            # * |QSYMM16        |QSYMM16        |
            # * |F16            |F16            |
            # * |F32            |F32            |
            r = re.search('\s*\*\s*Valid data type configurations:\s*', line)
            if r and state == States.DATA_LAYOUT_END:
                state = States.DATA_TYPE_START
                logging.debug(line)
                continue
            # Treat data configuration ends with a blank line only with " *"
            r = re.match('\s*\*\s*$', line)
            if r and state == States.DATA_TYPE_START:
                logging.debug(class_name)
                logging.debug(data_layout_list)
                logging.debug(io_list)
                logging.debug(data_type_list)
                class_no = class_no + 1
                if class_no > 1:
                    logging.debug(class_no)
                    self.generate_operator_cl_begin()
                self.generate_operator_dl_dt_info(class_name, data_layout_list, io_list, data_type_list)
                state = States.INIT
                data_type_list = []
                data_layout_list = []
                continue
            # Data type continues
            r = re.search('\s*\*(.*)', line)
            if r and state == States.DATA_TYPE_START:
                tmp = r.groups()[0]
                tmp = tmp.strip()
                if re.search('\|\:\-\-\-', tmp):
                    # Skip the table split row "|:-----"
                    continue
                else:
                    tmp = tmp.strip()
                    if re.search('.*(src|input|dst)', tmp):
                        io_list = tmp.split('|')
                    else:
                        data_type = tmp.split('|')
                        logging.debug(data_type)
                        data_type_list.append(data_type)
                    continue

        f.close()

    def generate_operator_cl_begin(self):
        self.xml += "<tr>\n"

    def generate_operator_common_info(self, class_name, operator_desc, nn_op_list, rowspan):
        tmp = "<tr>\n"
        # Store class name
        tmp += "  <td rowspan=\"" + rowspan + "\">" + class_name + "\n"
        tmp += "  <td rowspan=\"" + rowspan + "\" style=\"width:200px;\">" + operator_desc + "\n"
        tmp += "  <td rowspan=\"" + rowspan + "\">\n"
        tmp += "      <ul>\n"
        for item in nn_op_list:
            tmp += "       <li>"
            tmp += item.strip()
            tmp += "\n"
        tmp += "      </ul>\n"
        self.xml += tmp

    def generate_operator_dl_dt_info(self, class_name, data_layout, io_list, data_type_list):
        tmp = "  <td>" + class_name + "\n"
        # Store data layout info
        tmp += "  <td>\n"
        tmp += "      <ul>\n"
        for item in data_layout:
            tmp += "       <li>"
            tmp += item.strip()
            tmp += "\n"
        tmp += "      </ul>\n"
        tmp += "  <td>\n"
        # Store data type table
        tmp += "    <table>\n"
        tmp += "    <tr>"
        for io in io_list:
            # Make sure it's not empty string
            if len(io) != 0:
                tmp += "<th>"
                tmp += io.strip()
        tmp += "\n"
        for item in data_type_list:
            tmp += "    <tr>"
            for i in item:
                # Make sure it's not empty string
                if len(i) != 0:
                    tmp += "<td>"
                    tmp += i.strip()
            tmp += "\n"
        tmp += "    </table>\n"
        self.xml += tmp

    def generate_table_prefix(self):
        tmp = "<table>\n"
        tmp += "<caption id=\"multi_row\"></caption>\n"
        tmp += "<tr>\n"
        tmp += "  <th>Function\n"
        tmp += "  <th>Description\n"
        tmp += "  <th>Equivalent Android NNAPI Op\n"
        tmp += "  <th>Backends\n"
        tmp += "  <th>Data Layouts\n"
        tmp += "  <th>Data Types\n"
        self.xml += tmp

    def generate_table_ending(self):
        self.xml += "</table>\n"

    def dump_xml(self):
        print(self.xml)

    def update_dox_file(self):
        operator_list_dox = self.project_dir / "docs" / "user_guide" / "operator_list.dox"

        with open(operator_list_dox, "r") as f:
            dox_content = f.read()

        # Check that there is only one non-indented table (This table should be the operator list)
        x = re.findall("\n<table>", dox_content)
        y = re.findall("\n</table>", dox_content)
        if len(x) != 1 or len(y) != 1:
            raise RuntimeError("Invalid .dox file")

        repl_str = "\n" + self.xml[:-1]  # Extra / removed "\n" characters needed to make up for search regex
        new_file = re.sub("\n<table>(.|\n)*\n<\/table>", repl_str, dox_content)

        with open(operator_list_dox, "w") as f:
            f.write(new_file)
            print("Successfully updated operator_list.dox with the XML table of supported operators.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Updates the Compute Library documentation with a table of supported operators."
    )
    parser.add_argument(
        "--dump_xml",
        type=bool,
        default=False,
        required=False,
        help="Dump the supported operators table XML to stdout",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        required=False,
        help="Enables logging, helpful for debugging. Default: False",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(message)s", level=logging.DEBUG)

    table_xml = OperatorsTable()
    table_xml.generate_table_prefix()
    table_xml.generate_operator_list()
    table_xml.generate_table_ending()
    table_xml.update_dox_file()

    if args.dump_xml:
        table_xml.dump_xml()
