# StreamGC -- GPU-native streaming graph coloring
# Build system for algorithms/streamgc

NVCC        := nvcc
CUDA_STD    := --std=c++17
OPT_FLAGS   := -O3 --use_fast_math --generate-line-info
WARN_FLAGS  := -Xcompiler -Wall -Xcompiler -Wextra
DIAG_FLAGS  := -diag-suppress 20094
RDC_FLAGS   := -rdc=true
LINK_FLAGS  := -lcuda -lcudart -lpthread

INCLUDES    := -Iinclude/

# Source files for the main binary
SRCS := src/streamgc.cu \
        src/kernels/local_update.cu \
        src/kernels/boundary_update.cu \
        src/kernels/versioned_pull.cu \
        src/kernels/recolor.cu \
        src/kernels/conflict_repair.cu \
        src/kernels/epoch_maintenance.cu \
        src/host/graph_loader.cpp \
        src/host/partitioner.cpp \
        src/host/stream_gen.cpp \
        src/host/feeder_thread.cpp \
        src/host/query_server.cpp

TARGET := streaming_gc

# Default: multi-arch build
all: ARCH_FLAGS := -gencode arch=compute_80,code=sm_80 \
                   -gencode arch=compute_90,code=sm_90
all: $(TARGET)

# Fast: single-arch for current GPU
fast: ARCH_FLAGS := -arch=native
fast: $(TARGET)

friendster: ARCH_FLAGS := -gencode arch=compute_80,code=sm_80 \
                          -gencode arch=compute_90,code=sm_90 \
                          -DSTREAMGC_MAX_HISTOGRAM_COLORS=128 \
                          -DSTREAMGC_CSR_EXTRA_SLACK_PCT=50
friendster: $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ARCH_FLAGS) $(INCLUDES) $(SRCS) -o streaming_gc_friendster $(LINK_FLAGS)

friendster-tight: ARCH_FLAGS := -gencode arch=compute_80,code=sm_80 \
                                -gencode arch=compute_90,code=sm_90 \
                                -DSTREAMGC_MAX_HISTOGRAM_COLORS=64 \
                                -DSTREAMGC_CSR_EXTRA_SLACK_PCT=25
friendster-tight: $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ARCH_FLAGS) $(INCLUDES) $(SRCS) -o streaming_gc_friendster_tight $(LINK_FLAGS)

# ---- Ablation targets (§7.7). Each writes a separate binary. ----
ABL_FLAGS_BASE := -arch=native

abl-force-bitmap: ABL := -DSTREAMGC_ABLATION_FORCE_BITMAP
abl-force-bitmap: $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ABL_FLAGS_BASE) $(ABL) $(INCLUDES) $(SRCS) -o streaming_gc_abl_bitmap $(LINK_FLAGS)

abl-force-histogram: ABL := -DSTREAMGC_ABLATION_FORCE_HISTOGRAM
abl-force-histogram: $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ABL_FLAGS_BASE) $(ABL) $(INCLUDES) $(SRCS) -o streaming_gc_abl_histogram $(LINK_FLAGS)

abl-no-overflow: ABL := -DSTREAMGC_ABLATION_NO_OVERFLOW
abl-no-overflow: $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ABL_FLAGS_BASE) $(ABL) $(INCLUDES) $(SRCS) -o streaming_gc_abl_nooverflow $(LINK_FLAGS)

abl-live-deg: ABL := -DSTREAMGC_ABLATION_LIVE_DEG_PRIORITY
abl-live-deg: $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ABL_FLAGS_BASE) $(ABL) $(INCLUDES) $(SRCS) -o streaming_gc_abl_livedeg $(LINK_FLAGS)

ablations: abl-force-bitmap abl-force-histogram abl-no-overflow abl-live-deg

phi-instr: ARCH_FLAGS := -gencode arch=compute_80,code=sm_80 \
                         -gencode arch=compute_90,code=sm_90 \
                         -DSTREAMGC_INSTRUMENT_PHI
phi-instr: $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ARCH_FLAGS) $(INCLUDES) $(SRCS) -o streaming_gc_phi $(LINK_FLAGS)

# H200-only
h200: ARCH_FLAGS := -arch=sm_90 -DSTREAMGC_H200=1
h200: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(WARN_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(ARCH_FLAGS) $(INCLUDES) $(SRCS) -o $@ $(LINK_FLAGS)

# Unit test binaries
test_vertex_state: test/unit/test_vertex_state.cu include/vertex_state.cuh
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(ARCH_FLAGS) $(DIAG_FLAGS) $(INCLUDES) \
		$< -o $@ $(LINK_FLAGS)

test_priority: test/unit/test_priority.cu include/priority.cuh
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(ARCH_FLAGS) $(DIAG_FLAGS) $(INCLUDES) \
		$< -o $@ $(LINK_FLAGS)

test_color_select: test/unit/test_color_select.cu include/color_select.cuh include/streamgc.cuh
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(ARCH_FLAGS) $(DIAG_FLAGS) $(INCLUDES) \
		$< -o $@ $(LINK_FLAGS)

test_versioned_pull: test/unit/test_versioned_pull.cu include/streamgc.cuh
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(ARCH_FLAGS) $(DIAG_FLAGS) $(INCLUDES) \
		$< -o $@ $(LINK_FLAGS)

test_local_update: test/unit/test_local_update.cu include/streamgc.cuh include/color_select.cuh
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(ARCH_FLAGS) $(DIAG_FLAGS) $(INCLUDES) \
		$< -o $@ $(LINK_FLAGS)

# Integration test binaries
test_single_gpu: test/integration/test_single_gpu.cu $(SRCS)
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(ARCH_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(INCLUDES) $^ -o $@ $(LINK_FLAGS)

test_coloring_valid: test/integration/test_coloring_valid.cu include/streamgc.cuh include/color_select.cuh
	$(NVCC) $(CUDA_STD) $(OPT_FLAGS) $(ARCH_FLAGS) $(DIAG_FLAGS) $(RDC_FLAGS) \
		$(INCLUDES) $< -o $@ $(LINK_FLAGS)

# Run all unit tests
unit_tests: test_vertex_state test_priority test_color_select test_versioned_pull test_local_update
	@echo "=== Running all unit tests ==="
	./test_vertex_state
	./test_priority
	./test_color_select
	./test_versioned_pull
	./test_local_update

# Run all integration tests
integration_tests: test_single_gpu test_coloring_valid
	@echo "=== Running integration tests ==="
	./test_single_gpu
	./test_coloring_valid

clean:
	rm -f $(TARGET) test_vertex_state test_priority test_color_select test_versioned_pull test_local_update \
		test_single_gpu test_coloring_valid

.PHONY: all fast h200 friendster friendster-tight ablations abl-force-bitmap abl-force-histogram abl-no-overflow abl-live-deg phi-instr unit_tests integration_tests clean
