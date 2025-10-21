# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -Xcompiler -fPIC -m64
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I. -I.. -I../cuda-samples/Common -I../cuda-samples/Common/UtilNPP
LIBRARIES :=

# GST
INCLUDES  += -I/usr/include/gstreamer-1.0 -I/usr/include/glib-2.0 -I/usr/lib/x86_64-linux-gnu/glib-2.0/include -I/usr/include/x86_64-linux-gnu
LIBRARIES += -lgstcontroller-1.0 -lgstvideo-1.0 -lgstbase-1.0 -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0

ALL_CCFLAGS += --threads 0 -diag-suppress 1217
LIBRARIES += -lnppisu_static -lnppif_static -lnppc_static -lnppial_static

SAMPLE_ENABLED := 1

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

# Target rules
all: build

build: libgstmynppfilter.so

mynppfilter.o: mynppfilter.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

libgstmynppfilter.so: mynppfilter.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) --shared -o $@ $+ $(LIBRARIES)

clean:
	rm -rf .deps mynppfilter.o libgstmynppfilter.so
