CC = mpic++
CFLAGS = -Wall -Werror -pedantic -g -I./source
LD = mpic++
LDFLAGS = -g

# Define the project directories
BUILD_DIR = build
RESOURCES_DIR = resources

# Define target paths
MATRIX_BIN = $(BUILD_DIR)/im2col

all: $(MATRIX_BIN)

# Link im2col
$(MATRIX_BIN): $(BUILD_DIR)/im2col.o
	$(LD) $(LDFLAGS) -o $@ $^

# Compile im2col.o
$(BUILD_DIR)/im2col.o: ./source/im2col.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

N=1000
test: $(MATRIX_BIN)
	@echo testing on $(N)x$(N) matrices of random floats from 0-1
	@echo 1 process
	mpirun -np 1 $(MATRIX_BIN) $(N)
	@echo 2 processes
	mpirun -np 2 $(MATRIX_BIN) $(N)
	@echo 4 processes
	mpirun -np 4 $(MATRIX_BIN) $(N)
	
