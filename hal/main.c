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

// File             : main.c
// Author           : S. Nolting, A. Deshmukh
// Last Modified    : 14.08.2023
// Abstract         : AIRISC + tinyHLS demo program
//

#include <stdint.h>
#include <airisc.h>
#include <ee_printf.h>

/*#define CLOCK_HZ   (32000000) // processor clock frequency
#define UART0_BAUD (9600)     // default Baud rate
#define TICK_TIME  (CLOCK_HZ) // timer tick every 1s*/

volatile uint32_t flag = 0;


/**********************************************************************//**
 * Main program. Standard arguments (argc, argv) are not used as they are
 * all-zero anyway (cleared by crt0 start-up code).
 **************************************************************************/
int main(void) {

	uint32_t tmp = 0;
	//tmp |= (1 << IRQ_MTI); // machine timer interrupt
	tmp |= (1 << IRQ_XIRQ0) | (1 << IRQ_XIRQ1); // AIRISC-specific external interrupt channel 0 and 1
	tmp |= (1 << IRQ_XIRQ2) | (1 << IRQ_XIRQ3); // AIRISC-specific external interrupt channel 2 and 3
	tmp |= (1 << IRQ_XIRQ4) | (1 << IRQ_XIRQ5); // AIRISC-specific external interrupt channel 4 and 5
	tmp |= (1 << IRQ_XIRQ6) | (1 << IRQ_XIRQ7); // AIRISC-specific external interrupt channel 6 and 7
	cpu_csr_write(CSR_MIE, tmp); // enable interrupt sources


	uint32_t a,b,c,d;
	a = 0x00000000;
	b = 0x00400000;
	c = 0x00800000;
	d = 0x00C00000;

	for (int i = 0; i < 8; i++)
	{
		tinyhls_transmit_data(tinyhls, d);
		tinyhls_transmit_data(tinyhls, c);
		tinyhls_transmit_data(tinyhls, b);
		tinyhls_transmit_data(tinyhls, a);
	}

	tinyhls_start_accelerater(tinyhls);
	/*volatile uint32_t status = tinyhls_check_status(tinyhls);

	while(status != 0x00000002){
		status = tinyhls_check_status(tinyhls);
	}*/

	//volatile uint32_t output = tinyhls_receive_data(tinyhls);
	volatile uint32_t output;

	cpu_csr_set(CSR_MSTATUS, 1 << MSTATUS_MIE); // enable machine-mode interrupts
  
    
  while(1){

	  if(flag == 1) {
		  output = tinyhls_receive_data(tinyhls);
		  flag = 0;
	  }

  }
  

  return 0;
}

/**********************************************************************//**
 * Custom interrupt handler (overriding the default DUMMY handler from "airisc.c").
 *
 * @note This is a "normal" function - so NO 'interrupt' attribute!
 *
 * @param[in] cause Exception identifier from mcause CSR.
 * @param[in] epc Exception program counter from mepc CSR.
 **************************************************************************/
void interrupt_handler(uint32_t cause, uint32_t epc) {

  switch(cause) {

    // -------------------------------------------------------
    // Machine timer interrupt (RISC-V-specific)
    // -------------------------------------------------------
    case MCAUSE_TIMER_INT_M:

      // adjust timer compare register for next interrupt
      // this also clears/acknowledges the current machine timer interrupt
      //timer_set_timecmp(timer0, timer_get_time(timer0) + (uint64_t)TICK_TIME);

      //ee_printf("Uptime: %is\r\n", ++uptime);

      break;

    // -------------------------------------------------------
    // External interrupt (AIRISC-specific)
    // -------------------------------------------------------
    case MCAUSE_XIRQ0_INT:
    	flag = 1;
    case MCAUSE_XIRQ1_INT:
    case MCAUSE_XIRQ2_INT:
    case MCAUSE_XIRQ3_INT:
    case MCAUSE_XIRQ4_INT:
    case MCAUSE_XIRQ5_INT:
    case MCAUSE_XIRQ6_INT:
    case MCAUSE_XIRQ7_INT:
    case MCAUSE_XIRQ8_INT:
    case MCAUSE_XIRQ9_INT:
    case MCAUSE_XIRQ10_INT:
    case MCAUSE_XIRQ11_INT:
    case MCAUSE_XIRQ12_INT:
    case MCAUSE_XIRQ13_INT:
    case MCAUSE_XIRQ14_INT:
    case MCAUSE_XIRQ15_INT:

      // the lowest 4-bit of MCAUSE identify the actual XIRQ channel when MCAUSE == MCAUSE_XIRQ*_INT
      //ee_printf("External interrupt from channel %u\r\n", (cause & 0xf));

      // clear/acknowledge the current interrupt by clearing the according MIP bit
      cpu_csr_write(CSR_MIP, cpu_csr_read(CSR_MIP) & (~(1 << ((cause & 0xf) + IRQ_XIRQ0))));

      break;

    // -------------------------------------------------------
    // Invalid (not implemented) interrupt source
    // -------------------------------------------------------
    default:

      // invalid/unhandled interrupt - give debug information and halt
      //ee_printf("Unknown interrupt source! mcause=0x%08x epc=0x%08x\r\n", cause, epc);

      cpu_csr_write(CSR_MIE, 0); // disable all interrupt sources
      while(1); // halt and catch fire
  }

}




