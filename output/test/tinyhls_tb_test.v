/*
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
    //   \$$$$  |$$ |$$ |  $$ |\$$$$$$$ |$$ |  $$ |$$$$$$$$\$$$$$$  | 
    //    \____/ \__|\__|  \__| \____$$ |\__|  \__|\________|\______/ 
    //                         $$\   $$ | 
    //                         \$$$$$$  | 
    //                          \______/  
    //////////////////////////////////////////////////////////////////
    */

    

`timescale 1ns / 100ps

module tinyhls_tb();
	parameter BIT_WIDTH = 32;
	parameter x_len = 16;
	parameter o_len = 1;
	parameter in_chan = 2;
	reg [BIT_WIDTH*x_len*in_chan-1:0] X;
	wire [BIT_WIDTH*o_len-1:0] out_data;
	reg clk;
	reg reset_n;
	wire out_en;
	reg start;

	reg [31:0] a = 32'b0000_0000_0000_0000_0000_0000_0000_0000;
	reg [31:0] b = 32'b0000_0000_0100_0000_0000_0000_0000_0000;
	reg [31:0] c = 32'b0000_0000_1000_0000_0000_0000_0000_0000;
	reg [31:0] d = 32'b0000_0000_1100_0000_0000_0000_0000_0000;

	initial begin
		reset_n = 0;
		#(700) reset_n = 1;
	end

	initial begin
		clk = 0;
		forever #(50) clk = ~clk;
	end

	tinyhls_cnn cnn_inst (
	.X(X),
	.clk(clk),
	.reset_n(reset_n),
	.start_net(start),
	.output_data(out_data),
	.out_en(out_en)
	);

	initial begin
		X = {8{d,c,b,a}};
		start = 1'b1;
		#(1200)
		start = 1'b0;
		#(1000)
		wait(out_en);
		#(2000)
		$display("TB PASSED");
		$finish();
	end
endmodule