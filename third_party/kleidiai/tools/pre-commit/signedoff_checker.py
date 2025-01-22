#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
""" Checks that a patch has been signed-off """
import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger("SignedOff Checker")


class SignedOffByChecker:
    """
    A class that checks if a patch has been signed-off.

    Parameters
    ----------
    dir : 'str' (optional)
        the directory to run the checker on and check the latest patch
        Default: .
    """

    def __init__(self, dir=".") -> None:
        self.dir = os.path.abspath(dir)
        logger.debug(f"SignedOffBy checker is set up to run on {self.dir}")

    def run(self) -> None:
        """Runs the checker.

        Raises
        ------
        ValueError
            If the checker fails to execute
        """
        retval = 0
        logger.debug(f"Running SignOff checker on '{self.dir}'")
        try:
            os.chdir(self.dir)
            cmd = ["git", "show", "-s", "--format=%B", "HEAD"]
            commit_msg = subprocess.check_output(cmd).decode("utf-8")
            if "Signed-off-by:" not in commit_msg:
                logger.error(
                    f"Patch not signed-off (Could not find 'Signed-off-by:' in the commit message). Use: git commit -s"
                )
                retval = -1
            else:
                logger.info("Patch has been signed-off!")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"SignedOff checker failed with {e}")
            retval = -1

        if retval != 0:
            raise ValueError("SignedOffBy checker failed!")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        "-d",
        help="Folder to execute the checker in. Default: .",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--debug",
        "-D",
        help="Enable debug information. Default: False",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    return args


def run_signedoff_checker(args):
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger.debug(f"Arguments passed: {str(args.__dict__)}")

    retval = 0
    try:
        checker = SignedOffByChecker(args.dir)
        checker.run()
    except ValueError as e:
        logger.error("Exception caught in SignedOffBy checker: %s" % e)
        retval = 1

    return retval


if __name__ == "__main__":
    sys.exit(run_signedoff_checker(parse_arguments()))
