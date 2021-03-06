# BESTFITS
Bispectrum ESTimator FourIer Transform Shells

This code will compute the galaxy bisprectrum from galaxy survey data or mock catalogs. The main
goal is to create code that is robust and can be easily extended to handle many different data
file formats.

## Dependencies
This code has some external dependencies that you will need to manually resolve.
1. [FFTW version 3.3.5](http://fftw.org/)
2. [GSL version 2.4](https://www.gnu.org/software/gsl/)
3. [CCfits version 2.5](https://heasarc.gsfc.nasa.gov/fitsio/ccfits/)
4. [cfitsio version 3.430](https://heasarc.gsfc.nasa.gov/fitsio/)

The version numbers provided are the only ones that have been tested. It is possible that other
versions of the above libraries may still work, especially if there is only a minor version number 
difference.

## Code Structure
As much as possible, various functionalities have been factored out into separate files in a way that
makes sense. The factorization is not yet optimal, so later versions of the code may see more 
separation of functions into new files. The goal is to make updates to the code easy and allow for 
more direct testing of the individual components.

## TO DO
1. Test code and compare to results of GPU based code.
2. If results of first test agree, try using rectangular prism instead of cube to greatly reduce
memory requirements.
