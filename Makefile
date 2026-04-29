# Targets:
#   make            build the demo binary
#   make run        build and run the demo (slow: 100M iterations)
#   make test       build and run the unit tests (CI target)
#   make debug      build with -O0 -g for stepping through in a debugger
#   make vec-info   show which loops the compiler auto-vectorized (and which it didn't)
#   make asm        dump annotated assembly to main.s
#   make clean      remove built artifacts

CXX      := g++
SRC      := main.cpp
TEST_SRC := test.cpp
HDRS     := cfr.h
BIN      := main
TEST_BIN := test_cfr

# AVX2 + FMA covers all the intrinsics in cfr.h and is available on most CI
# runners. If your CPU/runner supports AVX-512 you can swap for:
#   -mavx512f -mavx512dq -mfma
CXXFLAGS := -O3 -std=c++17 -mavx2 -mfma -Wall -Wextra
LDFLAGS  :=

.PHONY: all run test debug vec-info asm clean

all: $(BIN)

$(BIN): $(SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)

$(TEST_BIN): $(TEST_SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) $(TEST_SRC) -o $@ $(LDFLAGS)

run: $(BIN)
	./$(BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

debug: CXXFLAGS := -O0 -g -std=c++17 -mavx2 -mfma -Wall -Wextra
debug: clean $(BIN)

vec-info: $(SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) -fopt-info-vec-optimized -fopt-info-vec-missed -c $(SRC) -o /dev/null

asm: $(SRC) $(HDRS)
	$(CXX) $(CXXFLAGS) -S -fverbose-asm $(SRC) -o $(BIN).s
	@echo "Assembly written to $(BIN).s"
	@echo "Try: grep -nE 'vmaxps|vfmadd|vhaddps|vmulps' $(BIN).s"

clean:
	rm -f $(BIN) $(TEST_BIN) $(BIN).s *.o
