#!/usr/bin/env python3

import glob
import os.path

eula_copyright = open("scripts/copyright_eula.txt",'r').read()

def add_cpp_copyright( f, content):
    global eula_copyright
    out = open(f,'w')
    out.write("/*\n")
    for line in eula_copyright.split('\n')[:-1]:
        out.write(" *");
        if line.strip() != "":
            out.write(" %s" %line)
        out.write("\n")
    out.write(" */\n")
    out.write(content.strip())
    out.write("\n")
    out.close()

def add_python_copyright( f, content):
    global eula_copyright
    out = open(f,'w')
    for line in eula_copyright.split('\n')[:-1]:
        out.write("#");
        if line.strip() != "":
            out.write(" %s" %line)
        out.write("\n")
    out.write(content.strip())
    out.write("\n")
    out.close()

def remove_comment( content ):
    comment=True
    out=""
    for line in content.split('\n'):
        if comment:
            if line.startswith(' */'):
                comment=False
            elif line.startswith('/*') or line.startswith(' *'):
                #print(line)
                continue
            else:
                raise Exception("ERROR: not a comment ? '%s'"% line)
        else:
            out += line + "\n"
    return out
def remove_comment_python( content ):
    comment=True
    out=""
    for line in content.split('\n'):
        if comment and line.startswith('#'):
            continue
        else:
            comment = False
            out += line + "\n"
    return out

for top in ['./arm_compute', './tests','./src','./examples','./utils/']:
    for root, _, files in os.walk(top):
        for f in files:
            path = os.path.join(root, f)

            if f in ['.clang-tidy', '.clang-format']:
                print("Skipping file: {}".format(path))
                continue

            with open(path, 'r', encoding='utf-8') as fd:
                content = fd.read()
                _, extension = os.path.splitext(f)

                if extension in ['.cpp', '.h', '.inl', '.cl']:
                    if not  content.startswith('/*'):
                        add_cpp_copyright(path, content)
                elif extension == '.py' or f in ['SConstruct', 'SConscript']:
                    if not content.startswith('# Copyright'):
                        add_python_copyright(path, content)
                elif f == 'CMakeLists.txt':
                    if not content.startswith('# Copyright'):
                        add_python_copyright(path, content)
                else:
                    raise Exception("Unhandled file: {}".format(path))
