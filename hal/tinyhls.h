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


// File          : tinyhls.h
// Author        : Ankur Deshmukh
// Creation Date : 14.08.2023
// Abstract      : Header for the tinyHLS accelerator.
//

#ifndef TINYHLS_H_
#define TINYHLS_H_

#include "defines.h"

void tinyhls_transmit_data(volatile TINYHLS_t* const handle, uint32_t data);
void tinyhls_start_accelerater(volatile TINYHLS_t* const handle);
uint32_t tinyhls_receive_data(volatile TINYHLS_t* const handle);
uint32_t tinyhls_check_status(volatile TINYHLS_t* const handle);

#endif


