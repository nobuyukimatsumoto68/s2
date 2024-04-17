CXX = g++-13 # icpx # g++
CXXFLAGS = -O3 -std=c++17 -fopenmp # -openmp
INCLUDES = -I/Users/nobuyukimatsumoto/grid/Grid/Eigen  # /projectnb/qfe/nmatsumo/opt/eigen/

# NVCC = nvcc
# NVCCFLAGS = -arch=sm_70 -O3 -lcusolver -std=c++17
# INCLUDES_CUDA =

DIR = ./

# # all: solve.o solve.o eps.o tt.o

# all: solve.o tt.o eps.o t_vev.o psipsi.o eig.o


geo.o: solve.cc header.hpp
	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@


# solve.o: solve.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp
# 	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) -o $(DIR)$@

# tt.o: tt_corr.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

# t_vev.o: t_vev.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

# psipsi.o: psipsi_corr.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

# eps.o: eps_corr.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@

# eigen_matrix.o: eigen_matrix.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@


# all: eig.o eigen_matrix.o

# eig.o: eig.cu header_cuda.hpp typedefs_cuda.hpp constants.hpp
# 	$(NVCC) $< $(NVCCFLAGS) $(INCLUDES_CUDA) -o $(DIR)$@

# eigen_matrix.o: eigen_matrix.cc header.hpp typedefs.hpp constants.hpp
# 	$(CXX) $< $(CXXFLAGS) $(INCLUDES) -o $(DIR)$@
