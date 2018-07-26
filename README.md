# BESTFITS
Bispectrum ESTimator FourIer Transform Shells

This code will compute the galaxy bisprectrum from galaxy survey data or mock catalogs. The main
goal is to create code that is robust and can be easily extended to handle many different data
file formats.

## Code Structure
As much as possible, various functionalities have been factored out into separate files in a way that
makes sense. The factorization is not yet optimal, so later versions of the code may see more 
separation of functions into new files. The goal is to make updates to the code easy and allow for 
more direct testing of the individual components.

## TO DO
1. Implement power spectrum calculation in order to enable calculation of shot noise.
2. Overload get_bispectrum function in shells.h to allow for inverse tranforms of shells to be
   stored in files.
3. Add function to file_io.h to output shell transforms to disk.
4. Add function to file_io.h to create memory mapped files for the shell transforms on disk.
