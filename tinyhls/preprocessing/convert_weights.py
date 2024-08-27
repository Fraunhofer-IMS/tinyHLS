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
#
# As a special exception, you may create a larger work that contains
# part or all of the tinyHLS hardware compiler and distribute that 
# work under the terms of your choice, so long as that work is not 
# itself a hardware compiler or template-based code generator or a 
# modified version thereof. Alternatively, if you modify or re-
# distribute the hardware compiler itself, you may (at your option) 
# remove this special exception, which will cause the hardware compi-
# ler and the resulting output files to be licensed under the GNU 
# General Public License without this special exception.
#
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

import os
import sys

__all__ = [
    "extract_weights",
    "convert_weights_to_hex",
    "convert_bias_to_hex",
    "create_verilog_includes"
]

# This function extracts the weights and biases from the model and outputs them as a .txt to the destination path. 
# Originally, a hls4ml function was used, but replaced by this function before the open source release. 
def extract_weights(model, destination_path):
    ctr = 0
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for i in range(len(model.layers)):
        if len(model.layers[i].weights) == 0:
            continue
        # print(i)
        weights = model.layers[i].weights[0].numpy().flatten()
        bias = model.layers[i].weights[1].numpy().flatten()
        # print(weights)
        # print(bias)
        w_str = ""
        b_str = ""
        for ind, j in enumerate(weights):
            if j == weights[-1] and ind == len(weights)-1:
                w_str = w_str + str(j)
            else:
                w_str = w_str + str(j) + "\n"
        for ind, j in enumerate(bias):
            if j == bias[-1] and ind == len(bias)-1:
                b_str = b_str + str(j)
            else:
                b_str = b_str + str(j) + "\n"
        # print(w_str)
        # print(b_str)
        f = open(destination_path + 'w' + str(ctr)+'.txt', 'w')
        f.write(w_str)
        f.close()
        f = open(destination_path + 'b' + str(ctr)+'.txt', 'w')
        f.write(b_str)
        f.close()
        ctr = ctr + 1

# This function converts the weights in the .txt from extract_weights to a .hex-file
def convert_weights_to_hex(source_path, destination_path, txt_files, BIT_WIDTH, INT_WIDTH):
    i = 0
    for f in txt_files:
        if os.path.isfile(os.path.join(source_path, f)):
            if 'b' in f:
                continue
            with open(os.path.join(source_path, "w" + str(i)+ ".txt")) as f_in, open(os.path.join(destination_path,"weights"+str(i)+".hex"), "w") as f_out:
                weights_str = f_in.read().split("\n")
                weights = [float(w) for w in weights_str]
                scale = 2**(BIT_WIDTH - INT_WIDTH)

                for w in weights:
                    w_fp = int(round(w * scale))
                    if BIT_WIDTH == 32:
                        f_out.write('{:08x}\n'.format(w_fp & 0xFFFFFFFF))  # Change bit width here
                    elif BIT_WIDTH == 24:
                        f_out.write('{:06x}\n'.format(w_fp & 0xFFFFFF))
                    elif BIT_WIDTH == 16:
                        f_out.write('{:04x}\n'.format(w_fp & 0xFFFF))
                    elif BIT_WIDTH == 8:
                        f_out.write('{:02x}\n'.format(w_fp & 0xFF))
                    else:
                        print("Unsupported Bit Width.. change and try again!")
                        sys.exit(0)
            i = i + 1
        else:
            print(f"The file {f} does not exist in the source directory.")


# This function converts the bias in the .txt from extract_weights to a .hex-file
def convert_bias_to_hex(source_path, destination_path, txt_files, BIT_WIDTH, INT_WIDTH):
    i = 0
    for f in txt_files:
        if os.path.isfile(os.path.join(source_path, f)):
            if 'w' in f:
                continue
            # shutil.copy2(os.path.join(source_path, f), os.path.join(copy_path, str(i)+"b.txt"))
            with open(os.path.join(source_path, "b" + str(i)+ ".txt"), "r") as f_in, open(os.path.join(destination_path,"bias"+str(i)+".hex"), "w") as f_out:
                weights_str = f_in.read().split("\n")
                weights = [float(w) for w in weights_str]
                scale = 2**(BIT_WIDTH - INT_WIDTH)

                for w in weights:
                    w_fp = int(round(w * scale))
                    if BIT_WIDTH == 32:
                        f_out.write('{:08x}\n'.format(w_fp & 0xFFFFFFFF))  # Change bit width here
                    elif BIT_WIDTH == 24:
                        f_out.write('{:06x}\n'.format(w_fp & 0xFFFFFF))
                    elif BIT_WIDTH == 16:
                        f_out.write('{:04x}\n'.format(w_fp & 0xFFFF))
                    elif BIT_WIDTH == 8:
                        f_out.write('{:02x}\n'.format(w_fp & 0xFF))
                    else:
                        print("Unsupported Bit Width.. change and try again!")
                        sys.exit(0)
            i = i + 1
        else:
            print(f"The file {f} does not exist in the source directory.")

#Create verilog files containing weight assignments for direct include
def create_verilog_includes(source_path, destination_path, model_json, BIT_WIDTH):
    trainable_count = 0
    for ctr, layer in enumerate(model_json["config"]["layers"]):
        layer_class = layer["class_name"]
        if layer_class == "Dense" or layer_class == "Conv1D":
            fw = 'weights'+str(trainable_count)+'.hex'
            fb = 'bias'+str(trainable_count)+'.hex'
            if os.path.isfile(os.path.join(source_path, fw)):
                f_in = open(os.path.join(source_path, fw), "r")
                name = os.path.splitext(fw)[0]
                f_out = open(os.path.join(destination_path, name+'.v'), "w")
                f_out.write("""/*
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
    //
    // As a special exception, you may create a larger work that contains
    // part or all of the tinyHLS hardware compiler and distribute that 
    // work under the terms of your choice, so long as that work is not 
    // itself a hardware compiler or template-based code generator or a 
    // modified version thereof. Alternatively, if you modify or re-
    // distribute the hardware compiler itself, you may (at your option) 
    // remove this special exception, which will cause the hardware compi-
    // ler and the resulting output files to be licensed under the GNU 
    // General Public License without this special exception.
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
    */\n""")
#                f_out.close()
                hex_values = []
                arr = f_in.read()
                i = 0
                while i < len(arr):
                    hex_values.append(arr[i:i+int(BIT_WIDTH/4)])
                    i = i+int(BIT_WIDTH/4)+1
                verilog_string = ""
                for i in range(len(hex_values)):
                    verilog_string = verilog_string + "assign mat" + str(ctr) + "[" + str(i) + "] = " + str(BIT_WIDTH) + "'h" + hex_values[i] + ";\n"
                f_out.write(verilog_string)
                f_in.close()
                f_out.close()
            if os.path.isfile(os.path.join(source_path, fb)):
                f_in = open(os.path.join(source_path, fb), "r")
                name = os.path.splitext(fb)[0]
                f_out = open(os.path.join(destination_path, name+'.v'), "w")
                hex_values = []
                arr = f_in.read()
                i = 0
                while i < len(arr):
                    hex_values.append(arr[i:i+int(BIT_WIDTH/4)])
                    i = i+int(BIT_WIDTH/4)+1
                verilog_string = ""
                for i in range(len(hex_values)):
                    verilog_string = verilog_string + "assign b" + str(ctr) + "[" + str(i) + "] = " + str(BIT_WIDTH) + "'h" + hex_values[i] + ";\n"
                f_out.write("""/*
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
    //
    // As a special exception, you may create a larger work that contains
    // part or all of the tinyHLS hardware compiler and distribute that 
    // work under the terms of your choice, so long as that work is not 
    // itself a hardware compiler or template-based code generator or a 
    // modified version thereof. Alternatively, if you modify or re-
    // distribute the hardware compiler itself, you may (at your option) 
    // remove this special exception, which will cause the hardware compi-
    // ler and the resulting output files to be licensed under the GNU 
    // General Public License without this special exception.
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
    */\n""")
                f_out.write(verilog_string)
                f_in.close()
                f_out.close()
            trainable_count = trainable_count + 1
