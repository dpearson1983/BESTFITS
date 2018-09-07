CXX = g++
CXXFLAGS = -fopenmp -march=bdver4 -mtune=bdver4 -O3
LIBFFTW = -lfftw3 -lfftw3_omp
LIBFITS = -lCCfits -lcfitsio
LIBGSL = -lgsl -lgslcblas -lm

build: cic cosmology file_io galaxy harppi power transformers shells line_of_sight bispec main.cpp
	$(CXX) $(LIBFFTW) $(LIBFITS) $(LIBGSL) $(CXXFLAGS) -o bestfits main.cpp obj/*.o
	
install: cic cosmology file_io galaxy harppi power transformers shells line_of_sight bispec main.cpp
	$(CXX) $(LIBFFTW) $(LIBFITS) $(LIBGSL) $(CXXFLAGS) -o $(HOME)/bin/bestfits main.cpp obj/*.o
	
cic: source/cic.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/cic.o source/cic.cpp
	
cosmology: source/cosmology.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(CXXFLAGS) -c -o obj/cosmology.o source/cosmology.cpp
	
file_io: source/file_io.cpp
	mkdir -p obj
	$(CXX) $(LIBFITS) $(LIBGSL) $(CXXFLAGS) -c -o obj/file_io.o source/file_io.cpp
	
galaxy: source/galaxy.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(CXXFLAGS) -c -o obj/galaxy.o source/galaxy.cpp
	
harppi: source/harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c -o obj/harppi.o source/harppi.cpp
	
power: source/power.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/power.o source/power.cpp
	
transformers: source/transformers.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/transformers.o source/transformers.cpp

shells: source/shells.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/shells.o source/shells.cpp
	
line_of_sight: source/line_of_sight.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/line_of_sight.o source/line_of_sight.cpp
	
bispec: source/bispec.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/bispec.o source/bispec.cpp
	
clean:
	rm obj/*.o
	rm bestfits
