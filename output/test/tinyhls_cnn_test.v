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
module tinyhls_cnn #(parameter BIT_WIDTH = 32, parameter FRAC_BITS = 22, parameter IN_LEN = 16, parameter OUT_LEN = 1, parameter NUM_IN_CHAN = 2) (
  input [BIT_WIDTH*NUM_IN_CHAN*IN_LEN-1:0] X,
  input start_net,
  output out_en,
  input clk,
  input reset_n,
  output [BIT_WIDTH*OUT_LEN-1:0] output_data
);
///////////////////////////////////////////////////Layer 1 : CONV LAYER///////////////////////////////////////////////////
reg [7:0] out1_index;
reg out_en_1;
parameter IN_CHAN1 = 2;
parameter OUT_CHAN1 = 6;
parameter o_len1 = 12;
parameter x_len1 = 16;
reg[7-1:0] j_1;
reg[3-1:0] k_1;
reg[2-1:0] c_1;
reg[3-1:0] d_1;
reg start_layer2;
parameter KERNEL_SIZE1 = 5;
parameter STRIDES1 = 1;
reg [BIT_WIDTH-1:0] X_temp1;
parameter ACTIV1 = 1;
reg [BIT_WIDTH-1:0] mat_temp1;
wire [BIT_WIDTH-1:0] mat1 [0:64-1];
wire [BIT_WIDTH-1:0] b1 [0:OUT_CHAN1-1];
reg [16:0] ctr; 
reg [BIT_WIDTH-1:0] FIFO0 [0:NUM_IN_CHAN*IN_LEN-1];
`include "weights/weights0.v"
`include "weights/bias0.v"
reg [BIT_WIDTH*2-1:0] sum1;
reg [BIT_WIDTH*2-1:0] sum1_reg;
reg [BIT_WIDTH-1:0] FIFO1 [0:128-1];
reg [7:0] pipeline1;
reg start_layer_d1;
///////////////////////////////////////////////////Layer 2 : MAXPOOL LAYER///////////////////////////////////////////////////
reg [5-1:0] out2_index;
reg [3-1:0] k_2;
reg [5-1:0] j_2;
parameter OUT_CHAN2 = 6;
parameter KERNEL_SIZE2 = 3;
parameter STRIDES2 = 3;
parameter IN_CHAN2 = 6;
parameter o_len2 = 4;
parameter x_len2 = 12;
reg out_en_2;
reg start_layer3;
reg [BIT_WIDTH-1:0] FIFO2 [0:32-1];
reg [BIT_WIDTH-1:0] outdata2;
reg [BIT_WIDTH-1:0] x_kernel2 [0:KERNEL_SIZE2-1];
reg [BIT_WIDTH-1:0] x_j2;
reg [5-1:0] pipeline2;
///////////////////////////////////////////////////Layer 3 : CONV LAYER///////////////////////////////////////////////////
reg [5:0] out3_index;
reg out_en_3;
parameter IN_CHAN3 = 6;
parameter OUT_CHAN3 = 12;
parameter o_len3 = 2;
parameter x_len3 = 4;
reg[5-1:0] j_3;
reg[4-1:0] k_3;
reg[3-1:0] c_3;
reg[2-1:0] d_3;
reg start_layer4;
parameter KERNEL_SIZE3 = 3;
parameter STRIDES3 = 1;
reg [BIT_WIDTH-1:0] X_temp3;
parameter ACTIV3 = 1;
reg [BIT_WIDTH-1:0] mat_temp3;
wire [BIT_WIDTH-1:0] mat3 [0:256-1];
wire [BIT_WIDTH-1:0] b3 [0:OUT_CHAN3-1];
`include "weights/weights1.v"
`include "weights/bias1.v"
reg [BIT_WIDTH*2-1:0] sum3;
reg [BIT_WIDTH*2-1:0] sum3_reg;
reg [BIT_WIDTH-1:0] FIFO3 [0:32-1];
reg [5:0] pipeline3;
reg start_layer_d3;
///////////////////////////////////////////////////Layer 4 : GAP LAYER///////////////////////////////////////////////////
reg [4-1:0] units4_index;
reg out_en_4;
parameter IN_CHAN4 = 12;
parameter out_len4 = 12;
reg start_layer5;
parameter x_len4 = 2;
reg [5-1:0] j_4;
reg [4-1:0] k_4;
reg [BIT_WIDTH-1:0] X_temp4;
reg [BIT_WIDTH+2-1:0] sum_gap;
reg [BIT_WIDTH-1:0] sum4 [0:16-1];
reg start_layer_d4;
reg [4:0] pipeline4;
reg [3-1:0] units5_index;
///////////////////////////////////////////////////Layer 5 : DENSE LAYER///////////////////////////////////////////////////
reg [7-1:0] i_5;
reg [7-1:0] j_5;
parameter ACTIV5 = 1;
parameter in_len5 = 12;
parameter out_len5 = 8;
reg start_layer6;
reg [BIT_WIDTH-1:0] X_temp5;
reg [FRAC_BITS+BIT_WIDTH-1:0] prod_temp5;
reg [BIT_WIDTH-1:0] sum5 [0:8-1];
reg [BIT_WIDTH-1:0] mat_temp5;
wire [BIT_WIDTH-1:0] mat5 [0:128-1];
wire [BIT_WIDTH-1:0] b5 [0:out_len5-1];
`include "weights/weights2.v"
`include "weights/bias2.v"
reg out_en_5;
reg [4-1:0] pipeline5;
reg [BIT_WIDTH-1:0] sum5_temp;
reg [1-1:0] units6_index;
///////////////////////////////////////////////////Layer 6 : DENSE LAYER///////////////////////////////////////////////////
reg [3-1:0] i_6;
reg [3-1:0] j_6;
parameter ACTIV6 = 0;
parameter in_len6 = 8;
parameter out_len6 = 1;
reg start_layer7;
reg [BIT_WIDTH-1:0] X_temp6;
reg [FRAC_BITS+BIT_WIDTH-1:0] prod_temp6;
reg [BIT_WIDTH-1:0] sum6 [0:1-1];
reg [BIT_WIDTH-1:0] mat_temp6;
wire [BIT_WIDTH-1:0] mat6 [0:8-1];
wire [BIT_WIDTH-1:0] b6 [0:out_len6-1];
`include "weights/weights3.v"
`include "weights/bias3.v"
reg out_en_6;
reg [3-1:0] pipeline6;
reg [BIT_WIDTH-1:0] sum6_temp;
assign out_en = start_layer7;
assign output_data[OUT_LEN*BIT_WIDTH-BIT_WIDTH*0-1:OUT_LEN*BIT_WIDTH-BIT_WIDTH*1] = sum6[0];
///////////////////////////////////////////////////Layer 1 : CONV LAYER///////////////////////////////////////////////////
always @ (*) begin
	for (ctr = 0; ctr < NUM_IN_CHAN*IN_LEN; ctr=ctr+1) begin 
		FIFO0[ctr] = X>>(ctr*BIT_WIDTH);
	end
end
always @(*) begin
	if (start_layer_d1) begin
		X_temp1 = FIFO0[d_1 + j_1 + x_len1*c_1];
		mat_temp1 = mat1[k_1 + (c_1*OUT_CHAN1) + (d_1*OUT_CHAN1*IN_CHAN1)];
		out1_index = j_1 + o_len1*k_1;
		sum1 = $signed(sum1_reg) + ($signed(X_temp1)*$signed(mat_temp1));
	end else begin
		X_temp1 = 0;
		mat_temp1 = 0;
		out1_index = 0;
		sum1 = sum1_reg;
	end
end
///////////////////////////////////////////////////Layer 2 : MAXPOOL LAYER///////////////////////////////////////////////////
always @(*) begin
	if (((pipeline1 >= KERNEL_SIZE2+j_2*STRIDES2) || pipeline1 == o_len1) && !out_en_2) begin
		out2_index = j_2 + k_2*o_len2;
		x_kernel2[0] = FIFO1[j_2*STRIDES2+0+k_2*o_len1];
		x_kernel2[1] = FIFO1[j_2*STRIDES2+1+k_2*o_len1];
		x_kernel2[2] = FIFO1[j_2*STRIDES2+2+k_2*o_len1];
		outdata2 = x_kernel2[0];
		for (x_j2 = 1; x_j2 < KERNEL_SIZE2; x_j2= x_j2 + 1) begin
			if($signed(x_kernel2[x_j2]) > $signed(outdata2)) begin 
				outdata2 = x_kernel2[x_j2];
			end
		end
	end else begin
		outdata2 = 0;
		out2_index = 0;
		x_j2 = 1;
		for (ctr = 0; ctr < KERNEL_SIZE2; ctr = ctr+1) begin
			x_kernel2[ctr] = 0;
		end
	end
end
///////////////////////////////////////////////////Layer 3 : CONV LAYER///////////////////////////////////////////////////
always @(*) begin
	if (pipeline2 >= KERNEL_SIZE3+j_3 || pipeline2 == o_len2-1) begin
		X_temp3 = FIFO2[d_3 + j_3 + x_len3*c_3];
		mat_temp3 = mat3[k_3 + (c_3*OUT_CHAN3) + (d_3*OUT_CHAN3*IN_CHAN3)];
		out3_index = j_3 + o_len3*k_3;
		sum3 = $signed(sum3_reg) + ($signed(X_temp3)*$signed(mat_temp3));
	end else begin
		X_temp3 = 0;
		mat_temp3 = 0;
		out3_index = 0;
		sum3 = sum3_reg;
	end
end
///////////////////////////////////////////////////Layer 4 : GAP LAYER///////////////////////////////////////////////////
always @ (*) begin
	if(start_layer_d4) begin
		X_temp4 = FIFO3[j_4+o_len3*k_4];
		units4_index = k_4;
	end else begin
		X_temp4 = 0;
		units4_index = 0;
	end
end
///////////////////////////////////////////////////Layer 5 : DENSE LAYER///////////////////////////////////////////////////
always @(*) begin
	if ((pipeline4 > j_5+1 || pipeline4 == out_len4) && !out_en_5) begin
		mat_temp5 = mat5[i_5+j_5*out_len5];
		X_temp5 = sum4[j_5];
		units5_index = i_5;
		prod_temp5 = $signed(X_temp5) * $signed(mat_temp5);
	end else begin
		mat_temp5 = 0;
		X_temp5 = 0;
		units5_index = 0;
		prod_temp5 = 0;
	end
end
///////////////////////////////////////////////////Layer 6 : DENSE LAYER///////////////////////////////////////////////////
always @(*) begin
	if ((pipeline5 > j_6+1 || pipeline5 == out_len5) && !out_en_6) begin
		mat_temp6 = mat6[i_6+j_6*out_len6];
		X_temp6 = sum5[j_6];
		units6_index = i_6;
		prod_temp6 = $signed(X_temp6) * $signed(mat_temp6);
	end else begin
		mat_temp6 = 0;
		X_temp6 = 0;
		units6_index = 0;
		prod_temp6 = 0;
	end
end
///////////////////////////////////////////////////Layer 1 : CONV LAYER///////////////////////////////////////////////////
always @ (posedge clk or negedge reset_n) begin
	if (!reset_n) begin
		out_en_1 <= 0;
		j_1 <= 0;
		k_1 <= 0;
		c_1 <= 0;
		d_1 <= 0;
		pipeline1 <= 0;
		sum1_reg <= 0;
		start_layer_d1 <= 0;
		start_layer2<= 0;
		for (ctr = 0; ctr < 128; ctr=ctr+1) begin
			FIFO1[ctr] <= 0;
		end
	end else begin
		if(start_net) begin
			start_layer_d1 <= 1'b1;
		end else begin
			start_layer_d1 <= start_layer_d1;
		end
		if(start_layer_d1) begin
			if(d_1 < KERNEL_SIZE1-1) begin
				d_1<= d_1 + 1;
				sum1_reg <= sum1;
			end else begin
				d_1 <= 0;
				sum1_reg <= sum1;
				if(c_1 < IN_CHAN1-1) begin
					c_1<= c_1 + 1;
				end else begin
					c_1 <= 0;
					sum1_reg <= sum1;
					if(ACTIV1) begin
						if($signed($signed(sum1[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b1[k_1])) > 0) begin
							FIFO1[out1_index] <= $signed(sum1[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b1[k_1]);
						end else begin
							FIFO1[out1_index] <= 0;
						end
					end else begin
						FIFO1[out1_index] <= $signed(sum1[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b1[k_1]);
					end
					if(k_1< OUT_CHAN1 - 1) begin
						k_1 <= k_1 + 1;
						sum1_reg <= 0; 
					end else begin
						k_1 <= 0;
						sum1_reg <= 0;
						pipeline1 <= pipeline1 + 1;
						if (j_1 < o_len1 - 1) begin
							j_1 <= j_1 + 1;
						end else begin
							out_en_1 <= 1'b1;
							start_layer2 <= ~start_layer2;
							start_layer_d1 <= 1'b0;
						end
					end
				end
			end
		end
	end
end
///////////////////////////////////////////////////Layer 2 : MAXPOOL LAYER///////////////////////////////////////////////////
always @ (posedge clk or negedge reset_n) begin
	if(!reset_n) begin
		k_2 <= 0;
		j_2 <= 0;
		out_en_2 <= 0;
		pipeline2 <= 0;
		start_layer3 <= 0;
		for(ctr = 0; ctr < 32; ctr = ctr+1) begin
			FIFO2[ctr] <= 0;
		end
	end else begin
		if (((pipeline1 >= KERNEL_SIZE2+j_2*STRIDES2) || pipeline1 == o_len1) && !out_en_2) begin
			FIFO2[out2_index] <= outdata2;
			if (k_2 < OUT_CHAN2-1) begin
				k_2 <= k_2 + 1;
			end else if (k_2 == OUT_CHAN2-1) begin
				j_2 <= j_2 + 1;
				pipeline2 <= pipeline2 + 1;
				k_2 <= 0;
			end
			if (k_2 == OUT_CHAN2-1 && j_2 == o_len2-1) begin
				out_en_2 <= 1'b1;
				start_layer3 <= ~start_layer3;
			end
		end
	end
end
///////////////////////////////////////////////////Layer 3 : CONV LAYER///////////////////////////////////////////////////
always @ (posedge clk or negedge reset_n) begin
	if (!reset_n) begin
		out_en_3 <= 0;
		j_3 <= 0;
		k_3 <= 0;
		c_3 <= 0;
		d_3 <= 0;
		pipeline3 <= 0;
		sum3_reg <= 0;
		start_layer_d3 <= 0;
		start_layer4<= 0;
		for (ctr = 0; ctr < 32; ctr=ctr+1) begin
			FIFO3[ctr] <= 0;
		end
	end else begin
		if ((pipeline2 >= KERNEL_SIZE3+j_3 || pipeline2 == o_len2) && !out_en_3) begin
			if(d_3 < KERNEL_SIZE3-1) begin
				d_3<= d_3 + 1;
				sum3_reg <= sum3;
			end else begin
				d_3 <= 0;
				sum3_reg <= sum3;
				if(c_3 < IN_CHAN3-1) begin
					c_3<= c_3 + 1;
				end else begin
					c_3 <= 0;
					sum3_reg <= sum3;
					if(ACTIV3) begin
						if($signed($signed(sum3[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b3[k_3])) > 0) begin
							FIFO3[out3_index] <= $signed(sum3[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b3[k_3]);
						end else begin
							FIFO3[out3_index] <= 0;
						end
					end else begin
						FIFO3[out3_index] <= $signed(sum3[BIT_WIDTH*2-1:BIT_WIDTH]<<10) + $signed(b3[k_3]);
					end
					if(k_3< OUT_CHAN3 - 1) begin
						k_3 <= k_3 + 1;
						sum3_reg <= 0; 
					end else begin
						k_3 <= 0;
						sum3_reg <= 0;
						pipeline3 <= pipeline3 + 1;
						if (j_3 < o_len3 - 1) begin
							j_3 <= j_3 + 1;
						end else begin
							out_en_3 <= 1'b1;
							start_layer4 <= ~start_layer4;
						end
					end
				end
			end
		end
	end
end
///////////////////////////////////////////////////Layer 4 : GAP LAYER///////////////////////////////////////////////////
always @ (posedge clk or negedge reset_n) begin
	if(!reset_n) begin
		j_4 <= 0;
		k_4 <= 0;
		pipeline4 <= 0;
		sum_gap <= 0;
		start_layer_d4 <= 0;
		start_layer5 <= 0;
		out_en_4 <= 0;
		for (ctr = 0; ctr < 16; ctr=ctr+1) begin
			sum4[ctr] <= 0;
		end
	end else begin
		if (start_layer4) begin
			start_layer_d4 <= 1'b1;
		end else begin
			start_layer_d4 <= start_layer_d4;
		end
		if (start_layer_d4) begin
			if (j_4 < x_len4-1) begin
				j_4 <= j_4 + 1;
				sum_gap <= $signed(X_temp4) + $signed(sum_gap);
			end else if (j_4 == x_len4-1) begin
				pipeline4 <=  pipeline4 + 1;
				sum4[units4_index] <= ($signed(sum_gap) + $signed(X_temp4)) / x_len4;
				sum_gap <= 0;
				if (k_4 < IN_CHAN4-1) begin 
					k_4 <= k_4 + 1;
					j_4 <= 0;
				end else begin
					start_layer_d4 <= 1'b0;
				end
			end
		end
	end
end
///////////////////////////////////////////////////Layer 5 : DENSE LAYER///////////////////////////////////////////////////
always @ (posedge clk or negedge reset_n) begin
	if(!reset_n) begin
		i_5 <= 0;
		j_5 <= 0;
		out_en_5 <= 0;
		pipeline5 <= 0;
		start_layer6 <= 0;
		sum5_temp <= 0;
		for (ctr = 0; ctr < 8; ctr=ctr+1) begin
			sum5[ctr] <= 0;
		end
	end else begin
		if ((pipeline4 > j_5+1 || pipeline4 == out_len4) && !out_en_5) begin
			if (j_5 < in_len5-1) begin
				j_5 <= j_5 + 1;
				sum5_temp <= $signed(sum5_temp) + $signed(prod_temp5[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]);
			end else if (j_5 == in_len5-1) begin
				pipeline5 <= pipeline5 + 1;
				j_5 <= 0;
				i_5 <= i_5 + 1;
				if (ACTIV5) begin
					if ($signed($signed(sum5_temp) + $signed(prod_temp5[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b5[i_5])) > 0) begin
						sum5[units5_index] <= $signed(sum5_temp) + $signed(prod_temp5[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b5[i_5]);
					end else begin
						sum5[units5_index] <= 0;
					end
				end else begin
					sum5[units5_index] <= $signed(sum5_temp) + $signed(prod_temp5[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b5[i_5]);
				end
				sum5_temp <= 0;
			end
			if (i_5 == out_len5-1 && j_5 == in_len5-1) begin
				start_layer6 <= 1'b1;
				out_en_5 <= 1'b1;
			end
		end
	end
end
///////////////////////////////////////////////////Layer 6 : DENSE LAYER///////////////////////////////////////////////////
always @ (posedge clk or negedge reset_n) begin
	if(!reset_n) begin
		i_6 <= 0;
		j_6 <= 0;
		out_en_6 <= 0;
		pipeline6 <= 0;
		start_layer7 <= 0;
		sum6_temp <= 0;
		for (ctr = 0; ctr < 1; ctr=ctr+1) begin
			sum6[ctr] <= 0;
		end
	end else begin
		if ((pipeline5 > j_6+1 || pipeline5 == out_len5) && !out_en_6) begin
			if (j_6 < in_len6-1) begin
				j_6 <= j_6 + 1;
				sum6_temp <= $signed(sum6_temp) + $signed(prod_temp6[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]);
			end else if (j_6 == in_len6-1) begin
				pipeline6 <= pipeline6 + 1;
				j_6 <= 0;
				i_6 <= i_6 + 1;
				if (ACTIV6) begin
					if ($signed($signed(sum6_temp) + $signed(prod_temp6[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b6[i_6])) > 0) begin
						sum6[units6_index] <= $signed(sum6_temp) + $signed(prod_temp6[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b6[i_6]);
					end else begin
						sum6[units6_index] <= 0;
					end
				end else begin
					sum6[units6_index] <= $signed(sum6_temp) + $signed(prod_temp6[FRAC_BITS+BIT_WIDTH-1:FRAC_BITS]) + $signed(b6[i_6]);
				end
				sum6_temp <= 0;
			end
			if (i_6 == out_len6-1 && j_6 == in_len6-1) begin
				start_layer7 <= 1'b1;
				out_en_6 <= 1'b1;
			end
		end
	end
end
endmodule
