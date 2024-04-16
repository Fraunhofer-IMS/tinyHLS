##################################################################
# tinyHLS Copyright (C) 2024 FRAUNHOFER INSTITUTE OF MICROELECTRONIC CIRCUITS AND SYSTEMS (IMS), DUISBURG, GERMANY. 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#   $$\     $$\                     $$\   $$\ $$\       $$$$$$\
#   $$ |    \__|                    $$ |  $$ |$$ |     $$  __$$\
# $$$$$$\   $$\ $$$$$$$\  $$\   $$\ $$ |  $$ |$$ |     $$ /  \__|
# \_$$  _|  $$ |$$  __$$\ $$ |  $$ |$$$$$$$$ |$$ |     \$$$$$$\
#   $$ |    $$ |$$ |  $$ |$$ |  $$ |$$  __$$ |$$ |      \____$$\
#   $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |     $$\   $$ |
#   \$$$$  |$$ |$$ |  $$ |\$$$$$$$ |$$ |  $$ |$$$$$$$$\\$$$$$$  |
#    \____/ \__|\__|  \__| \____$$ |\__|  \__|\________|\______/
#                         $$\   $$ |
#                         \$$$$$$  |
#                          \______/
###################################################################

# in_len:
#     for dense : number of input neurons
#     for others : number of channels

from math import floor
import os
import pathlib


network_hdl = [] #Contains all declarations
network_hdl1 = [] #Contains clocked statements
network_hdl2 = [] #Contains combinatoric statements 

__all__ = [
  "translate_model",
  "write_conv_input_resource_sync",
  "write_gap1D_sync",
  "write_maxpool_sync",
  "write_dense_resource_sync"
]

# This function generates the GAP 1D Layer. 
# The layer number is mostley needed for naming purposes in the code. 
# The input length describes the number of chanels, same for the out_units. 
# The test parameter is used if the layer is to be generated and tested as a standalone. 
def write_gap1D_sync(layer_number , in_len, filters, output_path=None, test=False):

  if test:
    network_hdl1.append("always @(posedge clk) begin\n")
    begin_hdl = ["module network #(parameter BIT_WIDTH = 32" + ", parameter IN_LEN = " + str(in_len) + ", parameter OUT_LEN = " + str(
    filters) + ", parameter NUM_IN_CHAN = " + str(filters) + ") (\n", "  input [BIT_WIDTH*NUM_IN_CHAN*IN_LEN-1:0] X,\n",
                "  input start_net,\n", "  output out_en,\n", "  input clk,\n", "  input reset_n,\n",
                "  output [BIT_WIDTH*OUT_LEN-1:0] output_data\n", ");\n"]
  i = layer_number
  n_res_bits = (filters - 1).bit_length()
  units_elem = pow(2, n_res_bits)

  n_in_bits = (in_len * filters - 1).bit_length()
  # network_hdl is the initializations and definitions. 
  network_hdl.append("///////////////////////////////////////////////////Layer " + str(i) + " : GAP LAYER///////////////////////////////////////////////////\n")
  network_hdl.append("reg [" + str(n_res_bits) + "-1:0] units" + str(i) + "_index;\n")
  network_hdl.append("reg out_en_" + str(i) + ";\n")
  network_hdl.append("parameter IN_CHAN" + str(i) + " = " + str(filters) + ";\n")
  network_hdl.append("parameter out_len" + str(i) + " = " + str(filters) + ";\n")
  network_hdl.append("reg start_layer" + str(i + 1) + ";\n")
  network_hdl.append("parameter x_len" + str(i) + " = " + str(in_len) + ";\n")
  network_hdl.append("reg [" + str(n_in_bits) + "-1:0] j_" + str(i) + ";\n")
  network_hdl.append("reg [" + str(n_res_bits) + "-1:0] k_" + str(i) + ";\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] X_temp" + str(i) + ";\n")
  network_hdl.append("reg [BIT_WIDTH+"+str(in_len)+"-1:0] sum_gap;\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] sum" + str(i) + " [0:" + str(units_elem) + "-1];\n")
  network_hdl.append("reg start_layer_d" + str(i) + ";\n")
  network_hdl.append("reg [" + str(n_res_bits) + ":0] pipeline" + str(i) + ";\n")
  # network_hdl2 is the combinatoric logic in always @(*) blocks. 
  network_hdl2.append("///////////////////////////////////////////////////Layer " + str(i) + " : GAP LAYER///////////////////////////////////////////////////\n")
  network_hdl2.append("always @ (*) begin\n")
  network_hdl2.append("\tif(start_layer_d" + str(i) + ") begin\n")
  network_hdl2.append("\t\tX_temp" + str(i) + " = FIFO" + str(i-1) + "[j_" + str(i) + "+o_len" + str(i-1) + "*k_" + str(i) + "];\n")
  network_hdl2.append("\t\tunits" + str(i) + "_index = k_" + str(i) + ";\n")
  network_hdl2.append("\tend else begin\n")
  network_hdl2.append("\t\tX_temp" + str(i) + " = 0;\n")
  network_hdl2.append("\t\tunits" + str(i) + "_index = 0;\n")
  network_hdl2.append("\tend\n")
  network_hdl2.append("end\n")
  # network_hdl1 is the sequential logic in always @ (posedge clk or negedge reset_n) blocks
  network_hdl1.append("///////////////////////////////////////////////////Layer " + str(i) + " : GAP LAYER///////////////////////////////////////////////////\n")
  network_hdl1.append("always @ (posedge clk or negedge reset_n) begin\n")
  network_hdl1.append("\tif(!reset_n) begin\n")
  network_hdl1.append("\t\tj_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tk_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tpipeline" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tsum_gap <= 0;\n")
  network_hdl1.append("\t\tstart_layer_d" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tstart_layer" + str(i+1) + " <= 0;\n")
  network_hdl1.append("\t\tout_en_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tfor (ctr = 0; ctr < " + str(units_elem) + "; ctr=ctr+1) begin\n")
  network_hdl1.append("\t\t\tsum" + str(i) + "[ctr] <= 0;\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend else begin\n")
  network_hdl1.append("\t\tif (start_layer" + str(i) + ") begin\n")
  network_hdl1.append("\t\t\tstart_layer_d" + str(i) + " <= 1'b1;\n")
  network_hdl1.append("\t\tend else begin\n")
  network_hdl1.append("\t\t\tstart_layer_d" + str(i) + " <= start_layer_d" + str(i) + ";\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\t\tif (start_layer_d" + str(i) + ") begin\n")
  network_hdl1.append("\t\t\tif (j_" + str(i) + " < x_len" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tj_" + str(i) + " <= j_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tsum_gap <= $signed(X_temp" + str(i) + ") + $signed(sum_gap);\n")
  network_hdl1.append("\t\t\tend else if (j_" + str(i) + " == x_len" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tpipeline" + str(i) + " <=  pipeline" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tsum" + str(i) + "[units" + str(i) + "_index] <= ($signed(sum_gap) + $signed(X_temp" + str(i) + ")) / x_len" + str(i) + ";\n")
  network_hdl1.append("\t\t\t\tsum_gap <= 0;\n")
  network_hdl1.append("\t\t\t\tif (k_" + str(i) + " < IN_CHAN" + str(i) + "-1) begin \n")
  network_hdl1.append("\t\t\t\t\tk_" + str(i) + " <= k_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\t\tj_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\tstart_layer_d" + str(i) + " <= 1'b0;\n")
  network_hdl1.append("\t\t\t\tend\n")
  network_hdl1.append("\t\t\tend\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend\n")
  network_hdl1.append("end\n")

  if test:
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    impl_path_output = output_path + "/gap1D_gen.v"
    # Logo and license 
    logo = """/*
    //////////////////////////////////////////////////////////////////
    // Copyright 2024 FRAUNHOFER INSTITUTE OF MICROELECTRONIC CIRCUITS AND SYSTEMS (IMS), DUISBURG, GERMANY.
    // --- All rights reserved --- 
    // SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
    // Licensed under the Solderpad Hardware License v 2.1 (the "License");
    // you may not use this file except in compliance with the License, or, at your option, the Apache License version 2.0.
    // You may obtain a copy of the License at
    // https://solderpad.org/licenses/SHL-2.1/
    // Unless required by applicable law or agreed to in writing, any work distributed under the License is distributed on an "AS IS" BASIS,
    // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    // See the License for the specific language governing permissions and limitations under the License.
    //
    //   $$\     $$\                     $$\   $$\ $$\       $$$$$$\ 
    //   $$ |    \__|                    $$ |  $$ |$$ |     $$  __$$\ 
    // $$$$$$\   $$\ $$$$$$$\  $$\   $$\ $$ |  $$ |$$ |     $$ /  \__| 
    // \_$$  _|  $$ |$$  __$$\ $$ |  $$ |$$$$$$$$ |$$ |     \$$$$$$\ 
    //   $$ |    $$ |$$ |  $$ |$$ |  $$ |$$  __$$ |$$ |      \____$$\ 
    //   $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |     $$\   $$ | 
    //   \$$$$  |$$ |$$ |  $$ |\$$$$$$$ |$$ |  $$ |$$$$$$$$\\$$$$$$  | 
    //    \____/ \__|\__|  \__| \____$$ |\__|  \__|\________|\______/ 
    //                         $$\   $$ | 
    //                         \$$$$$$  | 
    //                          \______/  
    //////////////////////////////////////////////////////////////////
    */\n"""
    with open(impl_path_output, 'w') as file:
      file.writelines(logo)
    with open(impl_path_output, 'a') as file:
      file.writelines(begin_hdl)
    with open(impl_path_output, 'a') as file:
      file.writelines(network_hdl)
    with open(impl_path_output, 'a') as file:
      file.writelines(network_hdl1)
    end_hdl = ["end\n", "endmodule\n"]
    with open(impl_path_output, 'a') as file:
      file.writelines(end_hdl)

# This function generates the Dense Layer. 
# The layer number is mostley needed for naming purposes in the code. 
# The input length describes the number of chanels, same for the out_units. 
# The weight and bias files have to be generated beforehand using the according script. 
# The num_in_chan is the bitwidth of each channel. 
# Kernel_size and strides come from the python model.
def write_dense_resource_sync(layer_number, in_len, out_units, WEIGHT_FILE, BIAS_FILE, is_last, activation_func, w_mat_size):

  i = layer_number
  n_bits = (w_mat_size - 1).bit_length()
  arr_elem = pow(2, n_bits)
  if n_bits == 0: n_bits = 1;
  n_res_bits = (out_units - 1).bit_length()

  units_elem = pow(2, n_res_bits)
  if n_res_bits == 0: n_res_bits = 1;
  #network_hdl is the initializations and signal definitions
  network_hdl.append("reg [" + str(n_res_bits) + "-1:0] units" + str(layer_number) + "_index;\n")

  n_res_bits = (in_len - 1).bit_length()
  if n_res_bits == 0: n_res_bits = 1;
  network_hdl.append("///////////////////////////////////////////////////Layer " + str(i) + " : DENSE LAYER///////////////////////////////////////////////////\n")
  network_hdl.append("reg [" + str(n_bits) + "-1:0] i_" + str(i) + ";\n")
  network_hdl.append("reg [" + str(n_bits) + "-1:0] j_" + str(i) + ";\n")
  network_hdl.append("parameter ACTIV" + str(i) + " = " + str(activation_func) + ";\n")
  network_hdl.append("parameter in_len" + str(i) + " = " + str(in_len) + ";\n")
  network_hdl.append("parameter out_len" + str(i) + " = " + str(out_units) + ";\n")
  network_hdl.append("reg start_layer" + str(i + 1) + ";\n")
  if i == 1:
    network_hdl.append("reg start_layer_d" + str(i) + ";\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] X_temp" + str(i) + ";\n")
  network_hdl.append("reg [FRAC_BITS+BIT_WIDTH-1:0] prod_temp" + str(i) + ";\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] sum" + str(i) + " [0:" + str(units_elem) + "-1];\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] mat_temp" + str(i) + ";\n")
  network_hdl.append("wire [BIT_WIDTH-1:0] mat" + str(i) + " [0:" + str(arr_elem) + "-1];\n")
  network_hdl.append("wire [BIT_WIDTH-1:0] b" + str(i) + " [0:out_len" + str(i) + "-1];\n")

  network_hdl.append("`include \"" + WEIGHT_FILE + "\"\n")
  network_hdl.append("`include \"" + BIAS_FILE + "\"\n")
  network_hdl.append("reg out_en_" + str(i) + ";\n")
  network_hdl.append("reg [" + str(n_res_bits) + "-1:0] pipeline" + str(i) + ";\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] sum" + str(i) + "_temp;\n")
  if is_last == 1:
    network_hdl.append("assign out_en = start_layer" + str(i + 1) + ";\n")
    u = 0
    for u in range(out_units):
      network_hdl.append(
        "assign output_data[OUT_LEN*BIT_WIDTH-BIT_WIDTH*" + str(u) + "-1:OUT_LEN*BIT_WIDTH-BIT_WIDTH*" + str(
          u + 1) + "] = sum" + str(i) + "[" + str(u) + "];\n")
  # network_hdl2 is the combinatoric logic in always @(*) blocks 
  network_hdl2.append("///////////////////////////////////////////////////Layer " + str(i) + " : DENSE LAYER///////////////////////////////////////////////////\n")
  network_hdl2.append("always @(*) begin\n")
  if i == 1:
    network_hdl2.append("\tif (start_layer_d" + str(i) + ") begin\n")
  else:
    network_hdl2.append("\tif ((pipeline" + str(i - 1) + " > j_" + str(i) + "+1 || pipeline" + str(i - 1) + " == out_len" + str(i - 1) + ") && !out_en_" + str(i) + ") begin\n")
  network_hdl2.append("\t\tmat_temp" + str(i) + " = mat" + str(i) + "[i_" + str(i) + "+j_" + str(i) + "*out_len" + str(i) + "];\n")
  network_hdl2.append("\t\tX_temp" + str(i) + " = sum" + str(i-1) + "[j_" + str(i) + "];\n")
  network_hdl2.append("\t\tunits" + str(i) + "_index = i_" + str(i) + ";\n")
  network_hdl2.append("\t\tprod_temp" + str(i) + " = $signed(X_temp" + str(i) + ") * $signed(mat_temp" + str(i) + ");\n")
  network_hdl2.append("\tend else begin\n")
  network_hdl2.append("\t\tmat_temp" + str(i) + " = 0;\n")
  network_hdl2.append("\t\tX_temp" + str(i) + " = 0;\n")
  network_hdl2.append("\t\tunits" + str(i) + "_index = 0;\n")
  network_hdl2.append("\t\tprod_temp" + str(i) + " = 0;\n")
  network_hdl2.append("\tend\n")
  network_hdl2.append("end\n")
  # network_hdl1 is the sequential logic in always @ (posedge clk or negedge reset_n) blocks
  network_hdl1.append("///////////////////////////////////////////////////Layer " + str(i) + " : DENSE LAYER///////////////////////////////////////////////////\n")    
  network_hdl1.append("always @ (posedge clk or negedge reset_n) begin\n")
  network_hdl1.append("\tif(!reset_n) begin\n")
  network_hdl1.append("\t\ti_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tj_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tout_en_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tpipeline" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tstart_layer" + str(i+1) + " <= 0;\n")
  network_hdl1.append("\t\tsum" + str(i) + "_temp <= 0;\n")
  network_hdl1.append("\t\tfor (ctr = 0; ctr < " + str(units_elem) + "; ctr=ctr+1) begin\n")
  network_hdl1.append("\t\t\tsum" + str(i) + "[ctr] <= 0;\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend else begin\n")

  if i == 1:
    network_hdl1.append("\t\tif (start_net) begin\n")
    network_hdl1.append("\t\t\tstart_layer_d" + str(i) + " <= 1'b1;\n")
    network_hdl1.append("\t\tend\n")

  if i == 1:
    network_hdl1.append("\t\tif (start_layer_d" + str(i) + ") begin\n")
  else:
    network_hdl1.append("\t\tif ((pipeline" + str(i - 1) + " > j_" + str(i) + "+1 || pipeline" + str(i - 1) + " == out_len" + str(i - 1) + ") && !out_en_" + str(i) + ") begin\n")
  network_hdl1.append("\t\t\tif (j_" + str(i) + " < in_len" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tj_" + str(i) + " <= j_" + str(i) + " + 1;\n")

  network_hdl1.append("\t\t\t\tsum" + str(i) + "_temp <= $signed(sum" + str(i) + "_temp) + $signed(prod_temp" + str(i) + "[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]);\n")
  network_hdl1.append("\t\t\tend else if (j_" + str(i) + " == in_len" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tpipeline" + str(i) + " <= pipeline" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tj_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\t\t\ti_" + str(i) + " <= i_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tif (ACTIV" + str(i) + ") begin\n")
  network_hdl1.append("\t\t\t\t\tif ($signed($signed(sum" + str(i) + "_temp) + $signed(prod_temp" + str(i) + "[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b" + str(i) + "[i_" + str(i) + "])) > 0) begin\n")
  network_hdl1.append("\t\t\t\t\t\tsum" + str(i) + "[units" + str(i) + "_index] <= $signed(sum" + str(i) + "_temp) + $signed(prod_temp" + str(i) + "[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b" + str(i) + "[i_" + str(i) + "]);\n")
  network_hdl1.append("\t\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\t\tsum" + str(i) + "[units" + str(i) + "_index] <= 0;\n")
  network_hdl1.append("\t\t\t\t\tend\n")
  network_hdl1.append("\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\tsum" + str(i) + "[units" + str(i) + "_index] <= $signed(sum" + str(i) + "_temp) + $signed(prod_temp" + str(i) + "[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b" + str(i) + "[i_" + str(i) + "]);\n")
  network_hdl1.append("\t\t\t\tend\n")
  network_hdl1.append("\t\t\t\tsum" + str(i) + "_temp <= 0;\n")
  network_hdl1.append("\t\t\tend\n")
  network_hdl1.append("\t\t\tif (i_" + str(i) + " == out_len" + str(i) + "-1 && j_" + str(i) + " == in_len" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tstart_layer" + str(i + 1) + " <= 1'b1;\n")
  if i == 1:
    network_hdl1.append("\t\t\t\tstart_layer_d" + str(i) + " <= 1'b0;\n")
  network_hdl1.append("\t\t\t\tout_en_" + str(i) + " <= 1'b1;\n")
  network_hdl1.append("\t\t\tend\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend\n")
  network_hdl1.append("end\n")


# This function generates the MaxPooling Layer. 
# The layer number is mostley needed for naming purposes in the code. 
# The input length describes the number of chanels, same for the out_units. 
# The num_in_chan is the bitwidth of each channel. 
# Kernel_size and strides come from the python model. 
def write_maxpool_sync(layer_number, in_len, out_units, num_in_chan,  kernel_size, strides):

  i = layer_number

  n_in_bits = (in_len * num_in_chan - 1).bit_length()

  n_out_bits = (out_units * num_in_chan - 1).bit_length()

  n_chan = num_in_chan.bit_length()

  out_elem = pow(2, n_out_bits)
  # network.hdl is the signal definition and parameter settings 
  network_hdl.append("///////////////////////////////////////////////////Layer " + str(i) + " : MAXPOOL LAYER///////////////////////////////////////////////////\n")
  network_hdl.append("reg [" + str(n_out_bits) + "-1:0] out" + str(i) + "_index;\n")
  network_hdl.append("reg [" + str(n_chan) + "-1:0] k_" + str(i) + ";\n")
  network_hdl.append("reg [" + str(n_out_bits) + "-1:0] j_" + str(i) + ";\n")
  
  network_hdl.append("parameter OUT_CHAN" + str(i) + " = " + str(num_in_chan) + ";\n")
  network_hdl.append("parameter KERNEL_SIZE" + str(i) + " = " + str(kernel_size) + ";\n")
  network_hdl.append("parameter STRIDES" + str(i) + " = " + str(strides) + ";\n")
  network_hdl.append("parameter IN_CHAN" + str(i) + " = " + str(num_in_chan) + ";\n")
  network_hdl.append("parameter o_len" + str(i) + " = " + str(out_units) + ";\n")
  network_hdl.append("parameter x_len" + str(i) + " = " + str(in_len) + ";\n")
  network_hdl.append("reg out_en_" + str(i) + ";\n")
  network_hdl.append("reg start_layer" + str(i + 1) + ";\n")
  
  network_hdl.append("reg [BIT_WIDTH-1:0] FIFO" + str(i) + " [0:" + str(out_elem) + "-1];\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] outdata" + str(i) + ";\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] x_kernel" + str(i) + " [0:KERNEL_SIZE" + str(i) + "-1];\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] x_j" + str(i) + ";\n")
  network_hdl.append("reg [" + str(n_out_bits) + "-1:0] pipeline" + str(i) + ";\n")
  # network_hdl2 is the purely combinatoric logic and always@(*) blocks 
  network_hdl2.append("///////////////////////////////////////////////////Layer " + str(i) + " : MAXPOOL LAYER///////////////////////////////////////////////////\n")
  network_hdl2.append("always @(*) begin\n")
  network_hdl2.append("\tif (((pipeline" + str(i - 1) + " >= KERNEL_SIZE" + str(i) + "+j_" + str(i) + "*STRIDES" + str(i) + ") || pipeline" + str(i - 1) + " == o_len" + str(i - 1) + ") && !out_en_" + str(i) + ") begin\n")
  network_hdl2.append("\t\tout" + str(i) + "_index = j_" + str(i) + " + k_" + str(i) + "*o_len" + str(i) + ";\n")
  for k in range(kernel_size):
    network_hdl2.append("\t\tx_kernel" + str(i) + "["  + str(k) + "] = FIFO" + str(i-1) + "[j_" + str(i) + "*STRIDES" + str(i) + "+" + str(k) + "+k_" + str(i) + "*o_len" + str(i-1) + "];\n")
  network_hdl2.append("\t\toutdata" + str(i) + " = x_kernel" + str(i) + "[0];\n")
  network_hdl2.append("\t\tfor (x_j" + str(i) + " = 1; x_j" + str(i) + " < KERNEL_SIZE" + str(i) + "; x_j" + str(i) + "= x_j" + str(i) + " + 1) begin\n")
  network_hdl2.append("\t\t\tif($signed(x_kernel" + str(i) + "[x_j" + str(i) + "]) > $signed(outdata" + str(i) + ")) begin \n")
  network_hdl2.append("\t\t\t\toutdata" + str(i) + " = x_kernel" + str(i) + "[x_j" + str(i) + "];\n")
  network_hdl2.append("\t\t\tend\n")
  network_hdl2.append("\t\tend\n")
  network_hdl2.append("\tend else begin\n")
  network_hdl2.append("\t\toutdata" + str(i) + " = 0;\n")
  network_hdl2.append("\t\tout" + str(i) + "_index = 0;\n")
  network_hdl2.append("\t\tx_j" + str(i) + " = 1;\n")
  network_hdl2.append("\t\tfor (ctr = 0; ctr < KERNEL_SIZE" + str(i) + "; ctr = ctr+1) begin\n")
  network_hdl2.append("\t\t\tx_kernel" + str(i) + "[ctr] = 0;\n")
  network_hdl2.append("\t\tend\n")
  network_hdl2.append("\tend\n")
  network_hdl2.append("end\n")
  # network_hdl1 is the sequential logic in the always @ (posedge clk or negedge reset_n) blocks
  network_hdl1.append("///////////////////////////////////////////////////Layer " + str(i) + " : MAXPOOL LAYER///////////////////////////////////////////////////\n")
  network_hdl1.append("always @ (posedge clk or negedge reset_n) begin\n")
  network_hdl1.append("\tif(!reset_n) begin\n")
  network_hdl1.append("\t\tk_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tj_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tout_en_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tpipeline" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tstart_layer" + str(i+1) + " <= 0;\n")
  network_hdl1.append("\t\tfor(ctr = 0; ctr < " + str(out_elem) + "; ctr = ctr+1) begin\n")
  network_hdl1.append("\t\t\tFIFO" + str(i) + "[ctr] <= 0;\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend else begin\n")
  network_hdl1.append("\t\tif (((pipeline" + str(i - 1) + " >= KERNEL_SIZE" + str(i) + "+j_" + str(i) + "*STRIDES" + str(i) + ") || pipeline" + str(i - 1) + " == o_len" + str(i - 1) + ") && !out_en_" + str(i) + ") begin\n")
  
  network_hdl1.append("\t\t\tFIFO" + str(i) + "[out" + str(i) + "_index] <= outdata" + str(i) + ";\n")
  
  network_hdl1.append("\t\t\tif (k_" + str(i) + " < OUT_CHAN" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tk_" + str(i) + " <= k_" + str(i) + " + 1;\n")

  network_hdl1.append("\t\t\tend else if (k_" + str(i) + " == OUT_CHAN" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tj_" + str(i) + " <= j_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tpipeline" + str(i) + " <= pipeline" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tk_" + str(i) + " <= 0;\n")

  network_hdl1.append("\t\t\tend\n")
  network_hdl1.append("\t\t\tif (k_" + str(i) + " == OUT_CHAN" + str(i) + "-1 && j_" + str(i) + " == o_len" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\tout_en_" + str(i) + " <= 1'b1;\n")
  network_hdl1.append("\t\t\t\tstart_layer" + str(i + 1) + " <= ~start_layer" + str(i + 1) + ";\n")
  
  network_hdl1.append("\t\t\tend\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend\n")
  network_hdl1.append("end\n")

# This function generates the 1D Convolution Layer. 
# The layer number is mostley needed for naming purposes in the code. 
# The input length describes the number of chanels, same for the out_units. 
# The weight and bias files have to be generated beforehand using the according script. 
# The num_in_chan is the bitwidth of each channel. 
# Kernel_size, strides, activation function and matrix size come from the python model.
# The fast parameter is introduced for latency (fast) or resource (!fast) optimization. 
# The test parameter is used if the layer is to be generated and tested as a standalone. 
def write_conv_input_resource_sync(layer_number, in_len, out_units, WEIGHT_FILE, BIAS_FILE, num_in_chan, filters,
                                    kernel_size, strides, activation_func, mat_size, fast, output_path=None, test=False):


  if test:
    network_hdl1.append("always @(posedge clk) begin\n")
    begin_hdl = ["module network #(parameter BIT_WIDTH = 32, parameter FRAC_BITS = 22" + ", parameter IN_LEN = " + str(in_len) + ", parameter OUT_LEN = " + str(
    out_units) + ", parameter NUM_IN_CHAN = " + str(num_in_chan) + ") (\n", "  input [BIT_WIDTH*NUM_IN_CHAN*IN_LEN-1:0] X,\n",
                "  input start_net,\n", "  output out_en,\n", "  input clk,\n", "  input reset_n,\n",
                "  output [BIT_WIDTH*OUT_LEN-1:0] output_data\n", ");\n"]
  i = layer_number

  n_mat_bits = (mat_size - 1).bit_length()
  mat_elem = pow(2, n_mat_bits)

  n_in_bits = (in_len * num_in_chan - 1).bit_length()

  n_out_bits = (out_units * filters - 1).bit_length()
  out_elem = pow(2, n_out_bits)

  n_in_chan = num_in_chan.bit_length()
  n_out_chan = filters.bit_length()
  n_kernel = kernel_size.bit_length()


  if fast:
    for ks in range(kernel_size):
      # network_hdl is the initializations and definitions. 
      network_hdl.append("reg [" + str(n_in_bits) + "-1:0] in" + str(i) + "_index_" + str(ks) + ";\n")
      network_hdl.append("reg [" + str(n_mat_bits) + "-1:0] max" + str(i) + "_index_" + str(ks) + ";\n")
  network_hdl.append("///////////////////////////////////////////////////Layer " + str(i) + " : CONV LAYER///////////////////////////////////////////////////\n")
  network_hdl.append("reg [" + str(n_out_bits) + ":0] out" + str(i) + "_index;\n")
  network_hdl.append("reg out_en_" + str(i) + ";\n")
  network_hdl.append("parameter IN_CHAN" + str(i) + " = " + str(num_in_chan) + ";\n")
  network_hdl.append("parameter OUT_CHAN" + str(i) + " = " + str(filters) + ";\n")
  network_hdl.append("parameter o_len" + str(i) + " = " + str(out_units) + ";\n")
  network_hdl.append("parameter x_len" + str(i) + " = " + str(in_len) + ";\n")
  network_hdl.append("reg[" + str(n_out_bits) + "-1:0] j_" + str(i) + ";\n")
  network_hdl.append("reg[" + str(n_out_chan) + "-1:0] k_" + str(i) + ";\n")
  network_hdl.append("reg[" + str(n_in_chan) + "-1:0] c_" + str(i) + ";\n")
  network_hdl.append("reg[" + str(n_kernel) + "-1:0] d_" + str(i) + ";\n")
  network_hdl.append("reg start_layer" + str(i + 1) + ";\n")

  network_hdl.append("parameter KERNEL_SIZE" + str(i) + " = " + str(kernel_size) + ";\n")
  network_hdl.append("parameter STRIDES" + str(i) + " = " + str(strides) + ";\n")
  if fast:
    network_hdl.append("reg [BIT_WIDTH-1:0] X_temp" + str(i) + " [0:" + str(kernel_size) + "-1];\n")
    network_hdl.append("reg [BIT_WIDTH-1:0] lin_sum" + str(i) + ";\n")
  else:
    network_hdl.append("reg [BIT_WIDTH-1:0] X_temp" + str(i) + ";\n")
  network_hdl.append("parameter ACTIV" + str(i) + " = " + str(activation_func) + ";\n")
  if fast:
    network_hdl.append("reg [BIT_WIDTH-1:0] mat_temp" + str(i) + " [0:" + str(kernel_size) + "-1];\n")
  else:
    network_hdl.append("reg [BIT_WIDTH-1:0] mat_temp" + str(i) + ";\n")

  network_hdl.append("wire [BIT_WIDTH-1:0] mat" + str(i) + " [0:" + str(mat_elem) + "-1];\n")
  network_hdl.append("wire [BIT_WIDTH-1:0] b" + str(i) + " [0:OUT_CHAN" + str(i) + "-1];\n")
  if i == 1:
    network_hdl.append("reg [16:0] ctr; \n")
    network_hdl.append("reg [BIT_WIDTH-1:0] FIFO" + str(i-1) + " [0:NUM_IN_CHAN*IN_LEN-1];\n")
  network_hdl.append("`include \""+ WEIGHT_FILE + "\"\n")
  network_hdl.append("`include \""+ BIAS_FILE + "\"\n")
  network_hdl.append("reg [BIT_WIDTH*2-1:0] sum" + str(i) + ";\n")
  network_hdl.append("reg [BIT_WIDTH*2-1:0] sum" + str(i) + "_reg;\n")
  network_hdl.append("reg [BIT_WIDTH-1:0] FIFO" + str(i) + " [0:" + str(out_elem) + "-1];\n")
  network_hdl.append("reg [" + str(n_out_bits) + ":0] pipeline" + str(i) + ";\n")
  if fast:
    network_hdl.append("reg [BIT_WIDTH+FRAC_BITS-1:0] outdata" + str(i) + " [0:" + str(kernel_size) + "-1];\n")
  # else:
  #   network_hdl.append("reg [BIT_WIDTH+FRAC_BITS-1:0] outdata" + str(i) + ";\n")
  network_hdl.append("reg start_layer_d" + str(i) + ";\n")
  if i == 1:
    # network_hdl2 is combinatoric logic in the always @(*) blocks
    network_hdl2.append("///////////////////////////////////////////////////Layer " + str(i) + " : CONV LAYER///////////////////////////////////////////////////\n")
    network_hdl2.append("always @ (*) begin\n")
    network_hdl2.append("\tfor (ctr = 0; ctr < NUM_IN_CHAN*IN_LEN; ctr=ctr+1) begin \n")
    network_hdl2.append("\t\tFIFO0[ctr] = X>>(ctr*BIT_WIDTH);\n")
    network_hdl2.append("\tend\n")
    network_hdl2.append("end\n")
  else:
    network_hdl2.append("///////////////////////////////////////////////////Layer " + str(i) + " : CONV LAYER///////////////////////////////////////////////////\n")
  network_hdl2.append("always @(*) begin\n")
  if i == 1:
    network_hdl2.append("\tif (start_layer_d" + str(i) + ") begin\n")
  else:
    network_hdl2.append(
      "\tif (pipeline" + str(i - 1) + " >= KERNEL_SIZE" + str(i) + "+j_" + str(i) + " || pipeline" + str(
        i - 1) + " == o_len" + str(i - 1) + "-1) begin\n")
  if fast:
    d = 0
    for d in range(kernel_size):
      network_hdl.append("        outdata" + str(i) + "[" + str(d) + "] = $signed(X_temp" + str(i) + "[" + str(
        d) + "]) * $signed(mat_temp" + str(i) + "[" + str(d) + "]);\n")
      if d == 1:
        network_hdl.append("        lin_sum" + str(i) + " = $signed(outdata" + str(
          i) + "[0][BIT_WIDTH+FRAC_BITS-1:FRAC_BITS]) + $signed(outdata" + str(
          i) + "[1][BIT_WIDTH+FRAC_BITS-1:FRAC_BITS]);\n")
      if d > 1:
        network_hdl.append(
          "        lin_sum" + str(i) + " = $signed(lin_sum" + str(i) + ") + $signed(outdata" + str(i) + "[" + str(
            d) + "][BIT_WIDTH+FRAC_BITS-1:FRAC_BITS]);\n")
  else:
    network_hdl2.append("\t\tX_temp" + str(i) + " = FIFO" + str(i-1) + "[d_" + str(i) + " + j_" + str(i) + " + x_len" + str(i) + "*c_" + str(i) + "];\n")
    network_hdl2.append("\t\tmat_temp" + str(i) + " = mat" + str(i) + "[k_" + str(i) + " + (c_" + str(i) + "*OUT_CHAN" + str(i) + ") + (d_" + str(i) + "*OUT_CHAN" + str(i) + "*IN_CHAN" + str(i) + ")];\n")
    network_hdl2.append("\t\tout" + str(i) + "_index = j_" + str(i) + " + o_len" + str(i) + "*k_" + str(i) + ";\n")
    network_hdl2.append("\t\tsum" + str(i) + " = $signed(sum" + str(i) + "_reg) + ($signed(X_temp" + str(i) + ")*$signed(mat_temp" + str(i) + "));\n")
    # network_hdl.append(
    #   "   outdata" + str(i) + " = $signed(X_temp" + str(i) + ") * $signed(mat_temp" + str(i) + ");\n")
  network_hdl2.append("\tend else begin\n")
  if fast:
    d = 0
    network_hdl.append("        lin_sum" + str(i) + " = 0;\n")
    for d in range(kernel_size):
      network_hdl.append("        outdata" + str(i) + "[" + str(d) + "] = 0;\n")
  else:
    network_hdl2.append("\t\tX_temp" + str(i) + " = 0;\n")
    network_hdl2.append("\t\tmat_temp" + str(i) + " = 0;\n")
    network_hdl2.append("\t\tout" + str(i) + "_index = 0;\n")
    network_hdl2.append("\t\tsum" + str(i) + " = sum" + str(i) + "_reg;\n")
  network_hdl2.append("\tend\n")
  network_hdl2.append("end\n")
  # network_hdl1 is the sequential logic in always @ (posedge clk or negedge reset_n)
  network_hdl1.append("///////////////////////////////////////////////////Layer " + str(i) + " : CONV LAYER///////////////////////////////////////////////////\n")
  network_hdl1.append("always @ (posedge clk or negedge reset_n) begin\n")
  network_hdl1.append("\tif (!reset_n) begin\n")
  network_hdl1.append("\t\tout_en_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tj_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tk_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tc_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\td_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tpipeline" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tsum" + str(i) + "_reg <= 0;\n")
  network_hdl1.append("\t\tstart_layer_d" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\tstart_layer" + str(i+1) + "<= 0;\n")
  network_hdl1.append("\t\tfor (ctr = 0; ctr < " + str(out_elem) + "; ctr=ctr+1) begin\n")
  network_hdl1.append("\t\t\tFIFO" + str(i) + "[ctr] <= 0;\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend else begin\n")
  if i == 1:
    network_hdl1.append("\t\tif(start_net) begin\n")
    network_hdl1.append("\t\t\tstart_layer_d" + str(i) + " <= 1'b1;\n")
    network_hdl1.append("\t\tend else begin\n")
    network_hdl1.append("\t\t\tstart_layer_d" + str(i) + " <= start_layer_d" + str(i) + ";\n")
    network_hdl1.append("\t\tend\n")
  
  if i == 1:
    network_hdl1.append("\t\tif(start_layer_d1) begin\n")
  else:
    network_hdl1.append("\t\tif ((pipeline" + str(i - 1) + " >= KERNEL_SIZE" + str(i) + "+j_" + str(i) + " || pipeline" + str(i - 1) + " == o_len" + str(i - 1) + ") && !out_en_" + str(i) + ") begin\n")

  network_hdl1.append("\t\t\tif(d_" + str(i) + " < KERNEL_SIZE" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\td_" + str(i) + "<= d_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tsum" + str(i) + "_reg <= sum" + str(i) + ";\n")
  network_hdl1.append("\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\td_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\t\t\tsum" + str(i) + "_reg <= sum" + str(i) + ";\n")
  network_hdl1.append("\t\t\t\tif(c_" + str(i) + " < IN_CHAN" + str(i) + "-1) begin\n")
  network_hdl1.append("\t\t\t\t\tc_" + str(i) + "<= c_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\tc_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\t\t\t\tsum" + str(i) + "_reg <= sum" + str(i) + ";\n")
  network_hdl1.append("\t\t\t\t\tif(ACTIV" + str(i) +") begin\n")
  network_hdl1.append("\t\t\t\t\t\tif($signed($signed(sum" + str(i) +"[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b" + str(i) + "[k_" + str(i) + "])) > 0) begin\n")
  network_hdl1.append("\t\t\t\t\t\t\tFIFO" + str(i) + "[out" + str(i) + "_index] <= $signed(sum" + str(i) +"[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b" + str(i) + "[k_" + str(i) + "]);\n")
  network_hdl1.append("\t\t\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\t\t\tFIFO" + str(i) + "[out" + str(i) + "_index] <= 0;\n")
  network_hdl1.append("\t\t\t\t\t\tend\n")
  network_hdl1.append("\t\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\t\tFIFO" + str(i) + "[out" + str(i) + "_index] <= $signed(sum" + str(i) +"[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b" + str(i) + "[k_" + str(i) + "]);\n")
  network_hdl1.append("\t\t\t\t\tend\n")
  network_hdl1.append("\t\t\t\t\tif(k_" + str(i) + "< OUT_CHAN" + str(i) + " - 1) begin\n")
  network_hdl1.append("\t\t\t\t\t\tk_" + str(i) + " <= k_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\t\t\tsum" + str(i) + "_reg <= 0; \n")
  network_hdl1.append("\t\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\t\tk_" + str(i) + " <= 0;\n")
  network_hdl1.append("\t\t\t\t\t\tsum" + str(i) + "_reg <= 0;\n")
  network_hdl1.append("\t\t\t\t\t\tpipeline" + str(i) + " <= pipeline" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\t\t\tif (j_" + str(i) + " < o_len" + str(i) + " - 1) begin\n")
  network_hdl1.append("\t\t\t\t\t\t\tj_" + str(i) + " <= j_" + str(i) + " + 1;\n")
  network_hdl1.append("\t\t\t\t\t\tend else begin\n")
  network_hdl1.append("\t\t\t\t\t\t\tout_en_" + str(i) + " <= 1'b1;\n")
  network_hdl1.append("\t\t\t\t\t\t\tstart_layer" + str(i+1) + " <= ~start_layer" + str(i + 1) + ";\n")
  if i == 1:
    network_hdl1.append("\t\t\t\t\t\t\tstart_layer_d1 <= 1'b0;\n")
  network_hdl1.append("\t\t\t\t\t\tend\n")
  network_hdl1.append("\t\t\t\t\tend\n")
  network_hdl1.append("\t\t\t\tend\n")
  network_hdl1.append("\t\t\tend\n")
  network_hdl1.append("\t\tend\n")
  network_hdl1.append("\tend\n")
  network_hdl1.append("end\n")

  if test:
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    impl_path_output = output_path + "/test.v"
    # Logo and License 
    logo = """/*
    //////////////////////////////////////////////////////////////////
    // Copyright 2024 FRAUNHOFER INSTITUTE OF MICROELECTRONIC CIRCUITS AND SYSTEMS (IMS), DUISBURG, GERMANY.
    // --- All rights reserved --- 
    // SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
    // Licensed under the Solderpad Hardware License v 2.1 (the "License");
    // you may not use this file except in compliance with the License, or, at your option, the Apache License version 2.0.
    // You may obtain a copy of the License at
    // https://solderpad.org/licenses/SHL-2.1/
    // Unless required by applicable law or agreed to in writing, any work distributed under the License is distributed on an "AS IS" BASIS,
    // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    // See the License for the specific language governing permissions and limitations under the License.
    //
    //   $$\     $$\                     $$\   $$\ $$\       $$$$$$\ 
    //   $$ |    \__|                    $$ |  $$ |$$ |     $$  __$$\ 
    // $$$$$$\   $$\ $$$$$$$\  $$\   $$\ $$ |  $$ |$$ |     $$ /  \__| 
    // \_$$  _|  $$ |$$  __$$\ $$ |  $$ |$$$$$$$$ |$$ |     \$$$$$$\ 
    //   $$ |    $$ |$$ |  $$ |$$ |  $$ |$$  __$$ |$$ |      \____$$\ 
    //   $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |     $$\   $$ | 
    //   \$$$$  |$$ |$$ |  $$ |\$$$$$$$ |$$ |  $$ |$$$$$$$$\\$$$$$$  | 
    //    \____/ \__|\__|  \__| \____$$ |\__|  \__|\________|\______/ 
    //                         $$\   $$ | 
    //                         \$$$$$$  | 
    //                          \______/  
    //////////////////////////////////////////////////////////////////
    */\n"""
    with open(impl_path_output, 'w') as file:
      file.writelines(logo)
    with open(impl_path_output, 'a') as file:
      file.writelines(begin_hdl)
    with open(impl_path_output, 'a') as file:
      file.writelines(network_hdl)
    with open(impl_path_output, 'a') as file:
      file.writelines(network_hdl1)
    end_hdl = ["end\n", "endmodule\n"]
    with open(impl_path_output, 'a') as file:
      file.writelines(end_hdl)



# This is the main function of this script. 
# It generates the HDL Code for all layers and then combines them in the output. 
# New Layers have to be considered in the if-Statements in this function! 
def translate_model(model_arch, param_path, output_path, quantization, file_name = "tinyhls_cnn", fast=False, use_relative_path=True):
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  impl_path_output = output_path + "/" + file_name + ".v"
  trainable_count = 0

  IN_LEN = model_arch["config"]["layers"][0]["config"]["batch_input_shape"][1]
  NUM_IN_CHAN = model_arch["config"]["layers"][0]["config"]["batch_input_shape"][2]
  OUT_LEN = model_arch["config"]["layers"][len(model_arch["config"]["layers"])-1]["config"]["units"]
  last_shape = [1, [IN_LEN, NUM_IN_CHAN]]
  # Includes License Agreement and Logo 
  begin_hdl = """/*
    //////////////////////////////////////////////////////////////////
    // Copyright 2024 FRAUNHOFER INSTITUTE OF MICROELECTRONIC CIRCUITS AND SYSTEMS (IMS), DUISBURG, GERMANY.
    // --- All rights reserved --- 
    // SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
    // Licensed under the Solderpad Hardware License v 2.1 (the "License");
    // you may not use this file except in compliance with the License, or, at your option, the Apache License version 2.0.
    // You may obtain a copy of the License at
    // https://solderpad.org/licenses/SHL-2.1/
    // Unless required by applicable law or agreed to in writing, any work distributed under the License is distributed on an "AS IS" BASIS,
    // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    // See the License for the specific language governing permissions and limitations under the License.
    //
    //   $$\     $$\                     $$\   $$\ $$\       $$$$$$\ 
    //   $$ |    \__|                    $$ |  $$ |$$ |     $$  __$$\ 
    // $$$$$$\   $$\ $$$$$$$\  $$\   $$\ $$ |  $$ |$$ |     $$ /  \__| 
    // \_$$  _|  $$ |$$  __$$\ $$ |  $$ |$$$$$$$$ |$$ |     \$$$$$$\ 
    //   $$ |    $$ |$$ |  $$ |$$ |  $$ |$$  __$$ |$$ |      \____$$\ 
    //   $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |     $$\   $$ | 
    //   \$$$$  |$$ |$$ |  $$ |\$$$$$$$ |$$ |  $$ |$$$$$$$$\\$$$$$$  | 
    //    \____/ \__|\__|  \__| \____$$ |\__|  \__|\________|\______/ 
    //                         $$\   $$ | 
    //                         \$$$$$$  | 
    //                          \______/  
    //////////////////////////////////////////////////////////////////
    */\n"""
  
  if use_relative_path:
    param_path = os.path.relpath(param_path, output_path)
  if model_arch["config"]["layers"][1]["class_name"] == "Dense":
    print("DENSE NETWORK")
    last_shape = [0, [IN_LEN, IN_LEN]]

  with open(impl_path_output, 'w') as file:
    file.write(begin_hdl)
  begin_hdl = ["module tinyhls_cnn #(parameter BIT_WIDTH = " + str(quantization['total']) + ", parameter FRAC_BITS = " + str(
    quantization['frac']) + ", parameter IN_LEN = " + str(IN_LEN) + ", parameter OUT_LEN = " + str(
    OUT_LEN) + ", parameter NUM_IN_CHAN = " + str(NUM_IN_CHAN) + ") (\n", "  input [BIT_WIDTH*NUM_IN_CHAN*IN_LEN-1:0] X,\n",
                "  input start_net,\n", "  output out_en,\n", "  input clk,\n", "  input reset_n,\n",
                "  output [BIT_WIDTH*OUT_LEN-1:0] output_data\n", ");\n"]
  with open(impl_path_output, 'a') as file:
    file.writelines(begin_hdl)

  # start translation for each layer 
  for i, layer in enumerate(model_arch["config"]["layers"]):
    layer_class = layer["class_name"]
    # Section for Dense Layer
    if layer_class == "Dense":
      print("dense")
      units = model_arch["config"]["layers"][i]["config"]['units']
      in_len = last_shape[1][1]
      activation = model_arch["config"]["layers"][i]["config"]['activation']
      last = 0
      if i == len(model_arch["config"]["layers"]) - 1:
        last = 1
      if i == 1: last = 2;
      ACTIV = 0;
      if activation == "relu":
        ACTIV = 1

      write_dense_resource_sync(i, in_len, units,
                                param_path + "/weights" + str(trainable_count) + ".v",
                                param_path + "/bias" + str(trainable_count) + ".v", last, ACTIV, in_len * units)
      trainable_count += 1
      last_shape[0] = 0
      last_shape[1][1] = units
    # Section for Conv1D Layer
    elif layer_class == "Conv1D":
      print("Conv1D")
      filters = model_arch["config"]["layers"][i]["config"]['filters']
      strides = model_arch["config"]["layers"][i]["config"]["strides"][0]
      in_len = last_shape[1][0]
      num_in_chan = last_shape[1][1]
      kernel_size = model_arch["config"]["layers"][i]["config"]["kernel_size"][0]
      units = floor((in_len - kernel_size) / strides + 1)

      activation = model_arch["config"]["layers"][i]["config"]['activation']
      WEIGHT_FILE = param_path + "/weights" + str(trainable_count) + ".v"
      BIAS_FILE = param_path + "/bias" + str(trainable_count) + ".v"
      ACTIV = 0;
      if activation == "relu":
        ACTIV = 1
      mat_size = filters * num_in_chan * kernel_size
      write_conv_input_resource_sync(i, in_len, units, WEIGHT_FILE, BIAS_FILE, num_in_chan, filters,
                                      kernel_size, strides, ACTIV, mat_size, fast)

      trainable_count += 1
      last_shape[0] = 1
      last_shape[1][0] = units
      last_shape[1][1] = filters
      
    #Section for GAP 1D Layer
    elif layer_class == "GlobalAveragePooling1D":
      print("gap1d")
      in_len = last_shape[1][0]
      filters = last_shape[1][1]
      write_gap1D_sync(i, in_len, filters)
      last_shape[1][1] = filters
      
    # Section for MaxPooling 1D Layer
    elif layer_class == "MaxPooling1D":
      print("maxpool1d")
      in_len = last_shape[1][0]
      filters = last_shape[1][1]
      num_in_chan = last_shape[1][1]
      kernel_size = model_arch["config"]["layers"][i]["config"]["pool_size"][0]
      strides = model_arch["config"]["layers"][i]["config"]["strides"][0]
      units = floor((in_len - kernel_size) / strides + 1)
      write_maxpool_sync(i, in_len, units, num_in_chan, kernel_size, strides)
      last_shape[1][0] = units
  end_hdl = ["endmodule\n"]
  
  # Output the HDL code in the file 
  # Initializations and Definitions 
  with open(impl_path_output, 'a') as file:
    file.writelines(network_hdl)
  # Combinatoric Logic 
  with open(impl_path_output, 'a') as file:
    file.writelines(network_hdl2)
  # Sequential Logic 
  with open(impl_path_output, 'a') as file:
    file.writelines(network_hdl1)
  # End of file (endmodule) 
  with open(impl_path_output, 'a') as file:
    file.writelines(end_hdl)
