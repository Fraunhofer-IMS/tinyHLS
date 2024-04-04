#!/usr/bin/env bash

set -e

cd $(dirname "$0")

TOP_DIR=${TOP_DIR:-../.}

cd "$TOP_DIR"/.ci

iverilog -v \
  -DCONFIG_IDEAL_SRAM_1 \
  -DSIM \
  -I "$TOP_DIR"/output/test \
  -I "$TOP_DIR"/output/test/weights \
  -o "$TOP_DIR"/.ci/sim \
  -c "$TOP_DIR"/.ci/sim_file_list.txt

vvp "$TOP_DIR"/.ci/sim
