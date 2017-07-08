# ==============================================================================
# MIT License

# Copyright (c) 2017 Brandon Bluemner

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
OUTPUT_DIR= bin
COMPILER = g++
FLAGS =-W -ggdb -std=c++11
SOURCE_DIR = source/
HEADERS_DIR = headers/

all: ReadWritePGMx driver house_holder_driver ReadWritePGM

driver: $(SOURCE_DIR)driver.cpp | make_dir
	$(COMPILER) $(FLAGS) $(HEADERS_DIR)matrix.hpp $(SOURCE_DIR)driver.cpp  -o $(OUTPUT_DIR)/driver.o
house_holder_driver: $(SOURCE_DIR)house_holder_driver.cpp | make_dir
	$(COMPILER) $(FLAGS) $(HEADERS_DIR)matrix.hpp $(SOURCE_DIR)house_holder_driver.cpp  -o $(OUTPUT_DIR)/house_holder_driver.o
ReadWritePGM: $(SOURCE_DIR)ReadWritePGM.cpp | make_dir
	$(COMPILER) $(FLAGS) $(HEADERS_DIR)matrix.hpp $(SOURCE_DIR)ReadWritePGM.cpp  -o $(OUTPUT_DIR)/ReadWritePGM.o
ReadWritePGMx: $(SOURCE_DIR)ReadWritePGMx.cpp | make_dir
	$(COMPILER) $(FLAGS) $(HEADERS_DIR)matrix.hpp $(SOURCE_DIR)ReadWritePGMx.cpp  -o $(OUTPUT_DIR)/ReadWritePGMx.o

make_dir:
	mkdir -p $(OUTPUT_DIR) | mkdir -p data
