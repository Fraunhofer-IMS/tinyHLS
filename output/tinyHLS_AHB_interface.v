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


/*
* This module can be used as a wrapper for the CNN generated by tinyHLS. 
* It is an AHB-Lite Interface, that can be used with microcontrollers such as the AIRISC. 
* Please compare the base address with your instanciation as well as your hal/defines.h!
*/
module tinyhls_ahb_interface #(parameter BASE_ADDR = 32'hC0000600, parameter BIT_WIDTH = 32, parameter IN_LEN = 16, parameter OUT_LEN = 1, parameter NUM_IN_CHAN = 2)
(
    input wire                             clk_i,
    input wire                             rst_ni,

    input       [32-1:0]                   haddr_i,
    input                                  hwrite_i,   
    input       [32-1:0]                   hsize_i,
    input       [2-1:0]                    htrans_i,
    input       [32-1:0]                   hwdata_i,  
    
    output reg  [32-1:0]                   hrdata_o,
    output                                 hready_o,
    output                                 hresp_o

);

    reg [BIT_WIDTH*NUM_IN_CHAN*IN_LEN-1:0] in_q;
    reg [BIT_WIDTH*NUM_IN_CHAN*IN_LEN-1:0] in_d;

    wire [BIT_WIDTH*OUT_LEN-1:0] output_data_d;

    reg                         hwrite_q;
    reg [32-1:0]                haddr_q;
    reg [32-1:0]                hrdata_d;
    
    reg [1:0]                       control_reg;
    reg [31:0]                      output_data_reg;
    reg [8:0]                       ctr_d, ctr_q;
    wire                            out_en_d;

    assign hready_o = 1'b1;
    assign hresp_o = 1'b0;

    //Sequential Logic 
    always @ (posedge clk_i or negedge rst_ni) begin
        if(~rst_ni) begin
            hwrite_q <= 1'b0;
            haddr_q <= 32'b0000000;
            hrdata_o <= 32'b00000000;
            in_q <= 0;
            output_data_reg <= 32'h00000000;
            ctr_q <= 0;
        end else begin
            hwrite_q <= hwrite_i;
            haddr_q <= haddr_i;
            hrdata_o <= hrdata_d;
            in_q <= in_d;
            ctr_q <= ctr_d;
            output_data_reg <= (output_data_d>>(BIT_WIDTH*ctr_d));
        end
    end


    //Combinatoric Logic 
    always @ (*) begin
        control_reg[0] = 1'b0;
        in_d = in_q;
        if (hwrite_q) begin
            case (haddr_q)
                (BASE_ADDR)     :   control_reg[0] = hwdata_i[0];
                (BASE_ADDR+4)   :   begin
                    in_d = {in_q[BIT_WIDTH*NUM_IN_CHAN*IN_LEN-1-BIT_WIDTH:0], hwdata_i};
                end
                default         : begin
                    control_reg[0] = 1'b0;
                    in_d = in_q;
                end
            endcase
        end

        hrdata_d = hrdata_o;
        if (|htrans_i) begin
            case (haddr_i)
                (BASE_ADDR)     :   hrdata_d    = control_reg;
                (BASE_ADDR+4)   :   hrdata_d    = output_data_reg;
                default         :   begin
                    hrdata_d = 32'h00000000;
                end   
            endcase
        end  
    end
    
    always @ (*) begin
	 ctr_d = ctr_q;
	 if ((|htrans_i) & (control_reg[1]==1'b1) & (haddr_i == (BASE_ADDR+4))) begin
	   ctr_d = ctr_d + 1;
	 end
	 control_reg[1] = out_en_d;
    end


    // This is the actual CNN that was converted and can be found in the output folder 
    tinyhls_cnn cnn_inst (
        .X(in_q),
        .clk(clk_i),
        .reset_n(rst_ni),
        .start_net(control_reg[0]),
        .output_data(output_data_d),
        .out_en(out_en_d)
	);

endmodule
