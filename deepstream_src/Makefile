CUDA_VER := 12.8
CUDA_PREFIX := /usr/local/cuda-$(CUDA_VER)
NVDS_VERSION := 8.0
NVDS_PREFIX := /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)

CXX := g++

SRCS := gstdsexample_cuda.cpp dsexample.cpp

INCS := $(wildcard *.h)
LIB :=libnvdsgst_mwdsexample.so

CXXFLAGS += -fPIC -DDS_VERSION=\"8.0.0\" \
	-I $(CUDA_PREFIX)/include \
	-I $(NVDS_PREFIX)/sources/includes


LIBS := -shared -Wl,-no-undefined \
	-L$(CUDA_PREFIX)/lib64 -Wl,-R$(CUDA_PREFIX)/lib64 \
    -lcudart -ldl -lnppc -lnppig -lnpps -lnppicc -lnppidei \
	-L$(NVDS_PREFIX)/lib -Wl,-R$(NVDS_PREFIX)/lib \
    -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface -lnvbufsurftransform\

OBJS := $(SRCS:.cpp=.o)

CXXFLAGS += $(shell pkg-config --cflags opencv412)

LIBS += -L/usr/local/lib/ -Wl,-R/usr/local/lib/ \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudaarithm

PKGS := gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0
CXXFLAGS += $(shell pkg-config --cflags $(PKGS))
LIBS += $(shell pkg-config --libs $(PKGS))

all: $(LIB)

build: $(LIB)

%.o: %.cpp $(INCS) Makefile
	@echo $(CXXFLAGS)
	$(CXX) -c -o $@ $(CXXFLAGS) $<

$(LIB): $(OBJS) Makefile
	@echo $(CXXFLAGS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(LIB)
