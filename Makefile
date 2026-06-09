CC = mpic++
CFLAGS = -Wall -Werror -pedantic -g -I./source
LD = mpic++
LDFLAGS = -g

# Define the project directories
BUILD_DIR = build
RESOURCES_DIR = resources

# Define target paths
MATRIX_BIN = $(BUILD_DIR)/im2col
NAIVE_MATRIX_BIN = $(BUILD_DIR)/naive_im2col

all: $(MATRIX_BIN) $(NAIVE_MATRIX_BIN)

# Link im2col
$(MATRIX_BIN): $(BUILD_DIR)/im2col.o
	$(LD) $(LDFLAGS) -o $@ $^

# Compile im2col.o
$(BUILD_DIR)/im2col.o: ./source/im2col.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Link naive im2col
$(NAIVE_MATRIX_BIN): $(BUILD_DIR)/naive_im2col.o
	$(LD) $(LDFLAGS) -o $@ $^

# Compile naive im2col.o
$(BUILD_DIR)/naive_im2col.o: ./source/naive_im2col.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Create the build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

M_WIDTH=1000
M_HEIGHT=1000
K_WIDTH=3
K_HEIGHT=3
STRIDE=1
test: $(MATRIX_BIN) $(NAIVE_MATRIX_BIN)
	@echo testing on $(M_WIDTH)x$(M_HEIGHT) matrices of random floats from 0-1
	@echo ~~~~~ 1 process ~~~~~
	mpirun -np 1 $(MATRIX_BIN) $(M_WIDTH) $(M_HEIGHT) $(K_WIDTH) $(K_HEIGHT) $(STRIDE) 
	mpirun -np 1 $(NAIVE_MATRIX_BIN) $(M_WIDTH) $(M_HEIGHT) $(K_WIDTH) $(K_HEIGHT) $(STRIDE) 
	@echo ~~~~~ 2 processes ~~~~~
	mpirun -np 2 $(MATRIX_BIN) $(M_WIDTH) $(M_HEIGHT) $(K_WIDTH) $(K_HEIGHT) $(STRIDE) 
	mpirun -np 2 $(NAIVE_MATRIX_BIN) $(M_WIDTH) $(M_HEIGHT) $(K_WIDTH) $(K_HEIGHT) $(STRIDE) 
	@echo ~~~~~ 4 processes ~~~~~
	mpirun -np 4 $(MATRIX_BIN) $(M_WIDTH) $(M_HEIGHT) $(K_WIDTH) $(K_HEIGHT) $(STRIDE) 
	mpirun -np 4 $(NAIVE_MATRIX_BIN) $(M_WIDTH) $(M_HEIGHT) $(K_WIDTH) $(K_HEIGHT) $(STRIDE) 
