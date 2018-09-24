CXX = g++
CXXFLAGS = -fopenmp -march=native -mtune=native -O3
CXXDEBUG = -fopenmp -march=native -mtune=native -g
LIBFFTW = -lfftw3 -lfftw3_omp
LIBFITS = -lCCfits -lcfitsio
LIBGSL = -lgsl -lgslcblas -lm

build: cic cosmology gadgetReader file_io galaxy harppi power transformers shells line_of_sight bispec main.cpp
	$(CXX) $(LIBFFTW) $(LIBFITS) $(LIBGSL) $(CXXFLAGS) -o bestfits main.cpp obj/*.o
	
debug_build: d_cic d_cosmology d_file_io d_galaxy d_harppi d_power d_transformers d_shells d_line_of_sight d_bispec main.cpp
	$(CXX) $(LIBFFTW) $(LIBFITS) $(LIBGSL) $(CXXDEBUG) -o $(HOME)/bin/bestfits_debug main.cpp obj/*.o
	
install: cic cosmology file_io galaxy harppi power transformers shells line_of_sight bispec main.cpp
	$(CXX) $(LIBFFTW) $(LIBFITS) $(LIBGSL) $(CXXFLAGS) -o $(HOME)/bin/bestfits main.cpp obj/*.o
	
cic: source/cic.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXFLAGS) -c -o obj/cic.o source/cic.cpp
	
cosmology: source/cosmology.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(CXXFLAGS) -c -o obj/cosmology.o source/cosmology.cpp
	
gadgetReader: source/gadgetReader.cpp
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -c -o obj/gadgetReader.o source/gadgetReader.cpp
	
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
	
d_cic: source/cic.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXDEBUG) -c -o obj/cic.o source/cic.cpp
	
d_cosmology: source/cosmology.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(CXXDEBUG) -c -o obj/cosmology.o source/cosmology.cpp
	
d_file_io: source/file_io.cpp
	mkdir -p obj
	$(CXX) $(LIBFITS) $(LIBGSL) $(CXXDEBUG) -c -o obj/file_io.o source/file_io.cpp
	
d_galaxy: source/galaxy.cpp
	mkdir -p obj
	$(CXX) $(LIBGSL) $(CXXDEBUG) -c -o obj/galaxy.o source/galaxy.cpp
	
d_harppi: source/harppi.cpp
	mkdir -p obj
	$(CXX) $(CXXDEBUG) -c -o obj/harppi.o source/harppi.cpp
	
d_power: source/power.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXDEBUG) -c -o obj/power.o source/power.cpp
	
d_transformers: source/transformers.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXDEBUG) -c -o obj/transformers.o source/transformers.cpp

d_shells: source/shells.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXDEBUG) -c -o obj/shells.o source/shells.cpp
	
d_line_of_sight: source/line_of_sight.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXDEBUG) -c -o obj/line_of_sight.o source/line_of_sight.cpp
	
d_bispec: source/bispec.cpp
	mkdir -p obj
	$(CXX) $(LIBFFTW) $(CXXDEBUG) -c -o obj/bispec.o source/bispec.cpp
