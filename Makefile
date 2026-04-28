# Targets:
#   make            build the binary
#   make run        build and run
#   make debug      build with -O0 -g for stepping through in a debugger
#   make vec-info   show which loops the compiler auto-vectorized (and which it didn't)
#   make asm        dump annotated assembly to cfr_simd_demo.s
#   make clean      remove built artifacts

CXX      := g++
SRC      := main.cpp
BIN      := main

# If your CPU supports AVX-512, you can swap the -m flags for
#   -mavx512f -mavx512dq -mfma
CXXFLAGS := -O3 -std=c++17 -mavx2 -mfma -Wall -Wextra
LDFLAGS  :=

.PHONY: all run debug vec-info asm clean

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

run: $(BIN)
	./$(BIN)

debug: CXXFLAGS := -O0 -g -std=c++17 -mavx2 -mfma -Wall -Wextra
debug: clean $(BIN)

# Print a report of every loop the compiler did or didn't vectorize.
vec-info: $(SRC)
	$(CXX) $(CXXFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c $< -o /dev/null

# Dump annotated assembly. search for vmaxps, vfmadd231ps, vhaddps :)
asm: $(SRC)
	$(CXX) $(CXXFLAGS) -S -fverbose-asm $< -o $(BIN).s
	@echo "Assembly written to $(BIN).s"
	@echo "Try: grep -nE 'vmaxps|vfmadd|vhaddps|vmulps' $(BIN).s"

clean:
	rm -f $(BIN) $(BIN).s
