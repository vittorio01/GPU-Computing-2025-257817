# Basic Folder Structure

This document outlines the basic structure of a typical project's folders and files.

## Contents

1. **Matrix directory**
   - `data/`: Contains all the matrices in the COO format from sparse.tamu.edu used for testing.
   **Libraries**
   - `libraries/`: Contains headers and implementation of used data structures.
    - `out/`: folder used as output for compiled libraries.
    - `src/`: source folder for libraries.
     - `dataLib/`: contains the implementation of Vector and Matrix data structures and utilities.
     - `mathStats/`: contains the implementation of statistical function for the expected value and variance.
   **paper source code**
   - `paper/`: The folder that contains the source code in latex for the paper.

   **algorithms**
   - `smm-csr-sequential/`: The implementation of the sequential algorithm for SpVm.
   - `smm-csr-parallel-1/`: The implementation of the first GPU algorithm for SpVm.
   - `smm-csr-parallel-2/`: The implementation of the second GPU algorithm for SpVm.
   - `smm-csr-parallel-3/`: The implementation of the third GPU algorithm for SpVm.
   - `smm-csr-parallel-4/`: The implementation of the fourth GPU algorithm for SpVm.

   **other files**
   - `flake.lock and flake.nix`: two files used for setting up the CUDA environment on NixOS. 
   - `readme.md`: This readme :) .

2. **Implementation folders**
   The folders `smm-csr-sequential` and `smm-csr-parallel-*` contains the various solution for the SpVm. These folders have the same structure:
   - `bin/`: the output folder that contains the compiled executable.
   - `out/`: the folder that contains the results stderr and stdout of the program when launched
   - `src/`: the folder that contains the source code   
   - `makefile`: used for compiling the binaries. 
   - `sbatch.sh`: the file that can be used for launching the program in a SLURM controlled server.  

   **a. Documentation**
   - `docs/`: Documentation files
     - `readme.md`: Main project documentation

   **b. Testing**
   - `tests/`: Test code and related files
     - `test_main.py`: Example test script

   **c. Build**
   - `build/`: Output from build process
     - `Makefile`: Build configuration file

3. **Files**

   - `LICENSE`: Project license
   - `README.md`: Main project documentation
   - `requirements.txt`: List of required packages
   - `setup.py`: Python package setup script

## Conclusion

This structure provides a clear organization for your project's files and makes it easier to manage and collaborate.