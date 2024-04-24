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
# As a special exception, you may create a larger work that contains part 
# or all of the tinyHLS hardware compiler and distribute that work under 
# the terms of the Solderpad Hardware License v2.1 (SHL-2.1), so long as 
# that work is not itself a hardware compiler or template-based code generator 
# or a modified version thereof. Alternatively, if you modify or redistribute 
# the hardware compiler itself, you may (at your option) remove this special 
# exception, which will cause the hardware compiler and the resulting output 
# files to be licensed under the GNU General Public License without this 
# special exception. 
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


import tensorflow as tf
import numpy as np
import os
import json
import re
import tinyhls


#  Basic model translation

train_samples = 10
num_elements = 16
num_in_chan = 2

x_train = np.ones((train_samples, num_elements, num_in_chan))
y_train = np.full((train_samples, 1), 500)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(filters=6, kernel_size=5, activation="relu", input_shape=(num_elements, num_in_chan), name="conv"))
model.add(tf.keras.layers.MaxPool1D(pool_size=3, name="maxpool"))
model.add(tf.keras.layers.Conv1D(filters=12, kernel_size=3, activation="relu", strides=1, name="conv2"))
model.add(tf.keras.layers.GlobalAveragePooling1D(name="gap"))
model.add(tf.keras.layers.Dense(units=8, activation="relu", name="dense"))
model.add(tf.keras.layers.Dense(units=1, activation="linear", name="dense1"))
model.summary()

optimizer = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(loss="mse", optimizer=optimizer, metrics='mae')
model.fit(x_train, y_train, epochs=2, batch_size=8)
model_json = model.to_json()
with open('./output/model_arch/modelTest.json', 'w') as f:
  json.dump(model_json, f)

with open('./output/model_arch/modelTest.json') as f:
  json_data = json.load(f)
json_str = json.loads(json_data)

x1 = np.tile(np.arange(0, 4, dtype=np.float32), 4)
x2 = np.tile(np.arange(0, 4, dtype=np.float32), 4)
# print(x[0:20])
# for xj, x2 in enumerate(x):
#   x[xj] = x2/pow(2, 10)
# print(x[0:10])
x = np.vstack((x1,x2)).T
x = np.expand_dims(x, axis=0)
# print(x.shape)

intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                                  outputs=model.get_layer('conv').output)

layer_output = intermediate_layer_model.predict(x)
print("keras conv output: \n", layer_output)

intermediate_layer_model = tf.keras.models.Model(inputs=model.input,
                                                  outputs=model.get_layer('conv2').output)

layer_output = intermediate_layer_model.predict(x)
print("keras conv2 output: \n", layer_output)

y = model.predict(x)
print("Prediction:", y)

BIT_WIDTH = 32
INT_WIDTH = 10

output_dir = './output/test/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

tinyhls.extract_weights(model, output_dir+'weights/')

path_temp = os.getcwd()
source_dir = os.path.join(path_temp, 'output/test/weights')   # replace this with the path to your directory
pattern = r'([wb])(\d+)\.txt'
files = os.listdir(source_dir)
txt_files = [f for f in files if f.endswith('.txt')]
hex_files = [f for f in files if f.endswith('.hex')]

txt_files.sort(key=lambda x: int(re.search(pattern, x).group(2)))
hex_files.sort()

tinyhls.convert_weights_to_hex(source_dir, source_dir, txt_files, BIT_WIDTH, INT_WIDTH)
tinyhls.convert_bias_to_hex(source_dir, source_dir, txt_files, BIT_WIDTH, INT_WIDTH)
tinyhls.create_verilog_includes(source_dir, source_dir, json_str, BIT_WIDTH)

quantization = {'total': 32, 'int': 10, 'frac': 22}

tinyhls.translate_model(model_arch=json_str, param_path= source_dir, output_path= output_dir, fast= False, quantization= quantization, file_name="tinyhls_cnn_test")

tinyhls.create_testbench(model_arch=json_str, quantization=quantization, clk_period=100, destination_path= output_dir, file_name='tinyhls_tb_test')

print("End of test.py; Translation done!")
