#!/usr/bin/env python3

# doit

from sys import executable, argv as sys_argv, exit as sys_exit
from os import environ
from pathlib import Path
import subprocess

from doit.action import CmdAction
from doit.cmd_base import ModuleTaskLoader
from doit.doit_cmd import DoitMain

DOIT_CONFIG = {"verbosity": 2, "action_string_formatting": "both"}

ROOT = Path(__file__).parent


def run_command(command, cwd=None):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, text=True, capture_output=True)
    print(f"stdout: {result.stdout}")
    print(f"stderr: {result.stderr}")
    result.check_returncode()

def task_Documentation():
    return {
        "actions": ["make -C docs {posargs}"],
        "doc": "Run a target in subdir 'doc'",
        "uptodate": [False],
        "pos_arg": "posargs",
    }

def task_DeployToGitHubPages():
    cwd = str(ROOT / "public")
    commands = [
        "git init",
        "cp .git/config ./docs/.git/config",
        "touch .nojekyll",
        "git add .",
        'git config --local user.email "push@gha"',
        'git config --local user.name "GHA"',
        "git commit -am '{posargs}'",
        "git push -u origin +HEAD:gh-pages",
    ]
    for command in commands:
        run_command(command, cwd=cwd)

    return {
        "actions": [
            CmdAction(lambda: run_command(command, cwd=cwd))
            for command in commands
        ],
        "doc": "Create a clean branch in subdir 'public' and push to branch 'gh-pages'",
        "pos_arg": "posargs",
    }


if __name__ == "__main__":
    sys_exit(DoitMain(ModuleTaskLoader(globals())).run(sys_argv[1:]))
