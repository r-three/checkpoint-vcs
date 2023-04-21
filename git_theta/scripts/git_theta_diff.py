import argparse
import sys
import textwrap

import numpy as np
from colorama import Fore, Style

from git_theta import checkpoints, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="git-theta diff program")
    parser.add_argument("path", help="path to file being diff-ed")

    parser.add_argument(
        "old_checkpoint", help="file that old version of checkpoint can be read from"
    )
    parser.add_argument("old_hex", help="SHA-1 hash of old version of checkpoint")
    parser.add_argument("old_mode", help="file mode for old version of checkpoint")

    parser.add_argument(
        "new_checkpoint", help="file that new version of checkpoint can be read from"
    )
    parser.add_argument("new_hex", help="SHA-1 hash of new version of checkpoint")
    parser.add_argument("new_mode", help="file mode for new version of checkpoint")

    args = parser.parse_args()
    return args


def color_string(s, color):
    return f"{color}{s}" if color else s


def bold_string(s):
    return f"{Style.BRIGHT}{s}"


def print_formatted(s, indent=0, color=None, bold=False):
    if indent:
        s = "\n".join(
            textwrap.wrap(
                s, indent=" " * 4 * indent, subsequent_indent=" " * 4 * (indent + 1)
            )
        )
    if color:
        s = color_string(s, color)
    if bold:
        s = bold_string(s)
    print(s)


def print_header(header, indent=0, color=None):
    print_formatted(header, indent=indent, color=color, bold=True)
    print_formatted("-" * len(header), indent=indent, color=color, bold=True)


def print_added_params_summary(added, indent=0, color=None):
    if added:
        print_header("ADDED PARAMETER GROUPS", indent=indent, color=color)
        for flattened_group, param in added.flatten().items():
            group = "/".join(flattened_group)
            print_formatted(group, indent=indent, color=color)
        print_formatted("\n")


def print_removed_params_summary(removed, indent=0, color=None):
    if removed:
        print_header("REMOVED PARAMETER GROUPS", indent=indent, color=color)
        for flattened_group, param in removed.flatten().items():
            group = "/".join(flattened_group)
            print_formatted(group, indent=indent, color=color)
        print_formatted("\n")


def print_modified_params_summary(modified, indent=0, color=None):
    if modified:
        print_header("MODIFIED PARAMETER GROUPS", indent=indent, color=color)
        for flattened_group, param in modified.flatten().items():
            group = "/".join(flattened_group)
            print_formatted(group, indent=indent, color=color)
        print_formatted("\n")


def main():
    args = parse_args()
    checkpoint_handler = checkpoints.get_checkpoint_handler()
    old_checkpoint = checkpoint_handler.from_file(args.old_checkpoint)
    new_checkpoint = checkpoint_handler.from_file(args.new_checkpoint)
    added, removed, modified = checkpoint_handler.diff(new_checkpoint, old_checkpoint)

    print_added_params_summary(added, indent=0, color=Fore.GREEN)
    print_removed_params_summary(removed, indent=0, color=Fore.RED)
    print_modified_params_summary(modified, indent=0, color=Fore.YELLOW)
