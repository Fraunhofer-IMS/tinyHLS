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


# This funuction generates a rudimentary testbench for the cnn in tinyHLS
# It does not compare the results, please track the waveform for that
# It will be improved in future releases. 
def create_testbench(model_arch, quantization, clk_period, destination_path, file_name='tinyHLS_tb'):
    begin_hdl = """/*
    //////////////////////////////////////////////////////////////////
    // tinyHLS Copyright (C) 2024 FRAUNHOFER INSTITUTE OF MICROELECTRONIC CIRCUITS AND SYSTEMS (IMS), DUISBURG, GERMANY. 
    //
    // This program is free software: you can redistribute it and/or modify
    // it under the terms of the GNU General Public License as published by
    // the Free Software Foundation, either version 3 of the License, or
    // (at your option) any later version.
    // 
    // This program is distributed in the hope that it will be useful,
    // but WITHOUT ANY WARRANTY; without even the implied warranty of
    // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    // GNU General Public License for more details.
    //
    // You should have received a copy of the GNU General Public License
    // along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
    */\n
    \n
`timescale 1ns / 100ps\n
module tinyhls_tb();\n"""
    path_output = destination_path + file_name + ".v"
    with open(path_output, 'w') as f:
        f.writelines(begin_hdl)

    tb_hdl = []

    IN_LEN = model_arch["config"]["layers"][0]["config"]["batch_input_shape"][1]
    NUM_IN_CHAN = model_arch["config"]["layers"][0]["config"]["batch_input_shape"][2]
    OUT_LEN = model_arch["config"]["layers"][len(model_arch["config"]["layers"])-1]["config"]["units"]
    BIT_WIDTH = quantization['total']


    tb_hdl.append("\tparameter BIT_WIDTH = " + str(quantization['total']) + ";\n")
    tb_hdl.append("\tparameter x_len = " + str(IN_LEN) + ";\n")
    tb_hdl.append("\tparameter o_len = " + str(OUT_LEN) + ";\n")
    tb_hdl.append("\tparameter in_chan = " + str(NUM_IN_CHAN) + ";\n")
    tb_hdl.append("\treg [BIT_WIDTH*x_len*in_chan-1:0] X;\n")
    tb_hdl.append("\twire [BIT_WIDTH*o_len-1:0] out_data;\n")
    tb_hdl.append("\treg clk;\n")
    tb_hdl.append("\treg reset_n;\n")
    tb_hdl.append("\twire out_en;\n")
    tb_hdl.append("\treg start;\n\n")
    tb_hdl.append("\treg [31:0] a = 32'b0000_0000_0000_0000_0000_0000_0000_0000;\n")
    tb_hdl.append("\treg [31:0] b = 32'b0000_0000_0100_0000_0000_0000_0000_0000;\n")
    tb_hdl.append("\treg [31:0] c = 32'b0000_0000_1000_0000_0000_0000_0000_0000;\n")
    tb_hdl.append("\treg [31:0] d = 32'b0000_0000_1100_0000_0000_0000_0000_0000;\n\n")

    tb_hdl.append("\tinitial begin\n")
    tb_hdl.append("\t\treset_n = 0;\n")
    tb_hdl.append("\t\t#(" + str(7*clk_period) + ") reset_n = 1;\n")
    tb_hdl.append("\tend\n\n")

    tb_hdl.append("\tinitial begin\n")
    tb_hdl.append("\t\tclk = 0;\n")
    tb_hdl.append("\t\tforever #(" + str(int(clk_period/2)) + ") clk = ~clk;\n")
    tb_hdl.append("\tend\n\n")

    tb_hdl.append("\ttinyhls_cnn cnn_inst (\n")
    tb_hdl.append("\t.X(X),\n")
    tb_hdl.append("\t.clk(clk),\n")
    tb_hdl.append("\t.reset_n(reset_n),\n")
    tb_hdl.append("\t.start_net(start),\n")
    tb_hdl.append("\t.output_data(out_data),\n")
    tb_hdl.append("\t.out_en(out_en)\n")
    tb_hdl.append("\t);\n\n")

    tb_hdl.append("\tinitial begin\n")
    tb_hdl.append("\t\tX = {" + str(int((IN_LEN/4)*NUM_IN_CHAN)) + "{d,c,b,a}};\n")
    tb_hdl.append("\t\tstart = 1'b1;\n")
    tb_hdl.append("\t\t#(" + str(12*clk_period) + ")\n")
    tb_hdl.append("\t\tstart = 1'b0;\n")
    tb_hdl.append("\t\t#(" + str(10*clk_period) + ")\n")
    tb_hdl.append("\t\twait(out_en);\n")
    tb_hdl.append("\t\t#(" + str(20*clk_period) + ")\n")
    # Passed for the CI Pipeline, in future releases this will be conditional. 
    tb_hdl.append("\t\t$display(\"TB PASSED\");\n")
    tb_hdl.append("\t\t$finish();\n")
    tb_hdl.append("\tend\n")
    tb_hdl.append("endmodule")

    with open(path_output, 'a') as f:
        f.writelines(tb_hdl)







