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

from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Hardware Compiler for tensorflow'
LONG_DESCRIPTION = 'Template based hardware compiler for tensorrflow, for creating efficient hardeware accelerators easily'

# Setting up
setup(
        name="tinyhls", 
        version=VERSION,
        author="Ankur Deshmukh",
        author_email="<ankur.ajay.deshmukh@ims.fraunhofer.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Linux :: Ubuntu",
        ]
)