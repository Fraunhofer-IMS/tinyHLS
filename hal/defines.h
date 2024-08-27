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


// File          : defines.h
// Author        : Ankur Deshmukh
// Creation Date : 14.08.2023
// Abstract      : Header for the tinyHLS accelerator.
//


#include <stdint.h>

// Struct for the accelerator
typedef struct
{
  uint32_t CTRL;          // TinyHLS control register
  uint32_t DATA;          // TinyHLS data register
} TINYHLS_t __attribute__((aligned(4)));

// Please compare with output/tinyHLS_AHB_interface.v or your instanciation for the base address! 
// Old (by OB.): 
// volatile static TINYHLS_t* const tinyhls = (TINYHLS_t *) 0xC000600; 
// New (by AD.): 
#define tinyhls (((volatile TINYHLS_t*) (0xC0000600)))
