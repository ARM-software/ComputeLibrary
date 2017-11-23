#!/usr/bin/env python3
# FIXME: Remove before the release

import os.path
import re
import sys

def process_comment(fd, comment, first_param, last_param):
    if first_param < 0:
        # Nothing to do: just copy the comment
        fd.write("".join(comment))
    else:
        params = list()

        # Measure the indentation of the first param and use that to create an empty comment line string:
        m = re.match(r" */", comment[0])

        if not m:
            raise Exception("{}: Not a comment ? '{}'".format(path,comment[first_param]))

        line_prefix = " " * len(m.group(0)) + "*"
        empty_line =  line_prefix +"\n"

        fd.write(comment[0])
        # Copy the non param lines with the correct indentation:
        for comment_line in range(1,first_param):
            line = comment[comment_line]
            m = re.match(" *\*(.*)", line)
            if not m:
                raise Exception("{}:{}: Not a comment line ? ".format(path, n_line - len(comment) + comment_line + 1))
            fd.write(line_prefix+ m.group(1)+"\n")

        # For each param split the line into 3 columns: param, param_name, description
        for param in range(first_param, last_param):
            m = re.match(r"[^@]+(@param\[[^\]]+\]) +(\S+) +(.+)", comment[param])

            if m:
                params.append( (" "+m.group(1), m.group(2), m.group(3)) )
            else:
                # If it's not a match then it must be a multi-line param description:

                m = re.match(" *\* +(.*)", comment[param])
                if not m:
                    raise Exception("{}:{}: Not a comment line ? ".format(path, n_line - len(comment) + param + 1))

                params.append( ("", "", m.group(1)) )

        # Now that we've got a list of params, find what is the longest string for each column:
        max_len = [0, 0]

        for p in params:
            for l in range(len(max_len)):
                max_len[l] = max(max_len[l], len(p[l]))

        # Insert an empty line if needed before the first param to make it easier to read:
        m = re.match(r" *\* *$", comment[first_param - 1])

        if not m:
            # insert empty line
            fd.write(empty_line)

        # Write out the formatted list of params:
        for p in params:
            fd.write("{}{}{} {}{} {}\n".format( line_prefix,
                    p[0], " " * (max_len[0] - len(p[0])),
                    p[1], " " * (max_len[1] - len(p[1])),
                    p[2]))

        # If the next line after the list of params is a command (@return, @note, @warning, etc), insert an empty line to separate it from the list of params
        if last_param < len(comment) - 1:
            if re.match(r" *\* *@\w+", comment[last_param]):
                # insert empty line
                fd.write(empty_line)

        # Copy the remaining of the comment with the correct indentation:
        for comment_line in range(last_param,len(comment)):
            line = comment[comment_line]
            m = re.match(" *\*(.*)", line)
            if not m:
                raise Exception("{}:{}: Not a comment line ? ".format(path, n_line - len(comment) + comment_line + 1))
            fd.write(line_prefix+ m.group(1)+"\n")

if __name__ == "__main__":
    n_file=0

    if len(sys.argv) == 1:
        paths = []

        for top_level in ["./arm_compute", "./src", "./examples", "./tests", "./utils", "./framework", "./support"]:
            for root, _, files in os.walk(top_level):
                paths.extend([os.path.join(root, f) for f in files])
    else:
        paths = sys.argv[1:]

    for path in paths:
        if (path[-3:] not in ("cpp", "inl") and
            path[-2:] not in ("cl") and
            path[-2:] not in ("cs") and
            path[-1] not in ("h")):
            continue

        print("[{}] {}".format(n_file, path))

        n_file += 1

        with open(path,'r+', encoding="utf-8") as fd:
            comment = list()
            first_param = -1
            last_param = -1
            n_line = 0

            lines = fd.readlines()
            fd.seek(0)
            fd.truncate()

            for line in lines:
                n_line += 1

                # Start comment
                # Match C-style comment /* anywhere in the line
                if re.search(r"/\*", line):
                    #print("Start comment {}".format(n_line))

                    if len(comment) > 0:
                        raise Exception("{}:{}: Already in a comment!".format(path,n_line))

                    comment.append(line)

                # Comment already started
                elif len(comment) > 0:
                    #print("Add line to comment {}".format(n_line))

                    comment.append(line)

                # Non-comment line
                else:
                    #print("Normal line {}".format(n_line))

                    fd.write(line)

                # Match param declaration in Doxygen comment
                # @param[in] name description
                if re.search(r"@param\[[^\]]+\] +\S+ +\S", line):
                    #print("Param {}".format(n_line))

                    if first_param < 0:
                        first_param = len(comment) - 1

                    last_param = len(comment)

                # Match end of C-style comment */
                if re.search(r"\*/", line):
                    if re.search('"[^"]*\*/[^"]*"', line):
                        #print("End of comment inside a string: ignoring")
                        pass
                    else:
                        #print("End comment {}".format(n_line))

                        if len(comment) < 1:
                            raise Exception("{}:{}: Was not in a comment! ".format(path, n_line))

                        #print("Process comment {} {}".format(first_param, last_param))

                        process_comment(fd, comment, first_param, last_param)

                        comment = list()
                        first_param = -1
                        last_param = -1
