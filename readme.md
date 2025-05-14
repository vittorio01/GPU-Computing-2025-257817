# Sparse Matrix and Vector multiplication algorithms
This repository contains the implementatin of various solution for a SpMV multiplication algorithm for CSR Sparse Matrix format on a Nvidia GPU on the global memory using only CUDA cores.
The main goal of this project is to find the best algorithm that is less memory bounded and performs better in terms of parallelization of the workload in difference situations. 

The results of this experiments and the description of the algorithms are documented in this dedicated paper located in `paper/` folder. The source code contains also small clarifications on the main steps of the programs. 

## Folder structure

   **Data folder**
   - `data/`: Contains all the matrices in the COO format from sparse.tamu.edu used for testing divided in base of the number of not null terms.

   **Libraries**
   - `libraries/`: Contains headers and implementation of used data structures.
     - `out/`: folder used as output for compiled libraries.
     - `src/`: source folder for libraries.
       - `dataLib/`: contains the implementation of Vector and Matrix data structures and utilities.
       - `mathStats/`: contains the implementation of statistical function for the expected value and variance.
         
   **Paper source code**
   - `paper/`: The folder that contains the source code in latex for the paper.
       - `main.tex`: The source code of the IEEE paper in latex.
       - `main.pdf`: The PDF generated.
       - `IEEEtran.cls and references.bib`: Latex files used for the IEEE template. 

   **Algorithms**
   - `smm-csr-sequential/`: The implementation of the sequential algorithm for SpMV multiplication.
   - `smm-csr-parallel-1/`: The implementation of the first GPU algorithm for SpMV multiplication.
   - `smm-csr-parallel-2/`: The implementation of the second GPU algorithm for SpMV multiplication.
   - `smm-csr-parallel-3/`: The implementation of the third GPU algorithm for SpMV multiplication.
   - `smm-csr-parallel-4/`: The implementation of the fourth GPU algorithm for SpMV multiplication.

   **Other files**
   - `flake.lock and flake.nix`: two files used for setting up the CUDA environment on NixOS. 
   - `readme.md`: This readme :) .


The folders `smm-csr-sequential` and `smm-csr-parallel-*` contains the various solution for the SpMV multiplication. These folders have the same structure:
   - `bin/`: the output folder that contains the compiled executable.
   - `out/`: the folder that contains the results stderr and stdout of the program when launched
   - `src/`: the folder that contains the source code   
   - `makefile`: used for compiling the binaries. 
   - `sbatch.sh`: the file that can be used for launching the program in a SLURM controlled server.  


## Build requirements
To compile the algorithm is necessary Linux machine with a preinstalled version of the CUDA SDK >= 12.5 (during the development I used my Nvidia RTX 4050 Laptop with the 12.8 version but all tests are performed on 12.5 version) and other C/C++ standard tools:
- `gnumake`
- `CUDA nvcc`
- `CUDA cudart`
- `gcc`

For NixOS users there is a `flake.nix` file that already contains the entire necessary environments for the development. In this case the main requirement is to have a preinstalled version of the CUDA sdk (because otherwise there is a strong possibility to have problems during the runtime). 

## How to compile the algorithms ##
To launch the environment on NixOS is necessary first to launch this command (On other linux distribution it is not necessary): 

```
nix develop
```

To compile a specific algorithm there is a dedicated `makefile` located in the folders `smm-csr-*` that automatically compiles the necessary custom libraries and the source code. 
```
makefile
```
The generated executable has the same name of the directory and will be located in the `bin` folder (the makefile creates it automatically). 

## How to launch the algorithms ##

The compiled executable can be launched directly on a Linux platform with a Nvidia card and a CUDA runtime with the same version of the SDK. The program requires also three different arguments:
- The directory of the input matrix in a COO sparse.tamu.edu format (the program will also convert the matrix in its CSR format).
- `-b "blocks"`: the number of blocks of the GPU to use.
- `-t "threads"`: the number of threads per block of the GPU to use.

For example, to launch the first algorithm using one of the matrices in the `data/` folder using 10 blocks and 256 threads:
```
./bin/smm-csr-parallel-1 ../data/lab_test_matrix.mtx -b 10 -t 256
```
In the same folder there is also a file `sbatch.sh` for launching the executable in a SLURM server. The arguments' format remains the same:
```
sbatch sbatch.sh ../data/lab/text/matrix.mtx -b 10 -t 256
```
The listing of the STDOUT and STDERR will be placed in the `out/` folder with the following format: `output_"job number".out` and `error_"job_number".err`.

## Credits ##

Suite Sparse Matrix Collection (https://sparse.tamu.edu/)

![alt text][logo]

[logo]: https://www.unitn.it/themes/custom/unitn_eventi/logointerno.svg "Università degli studi di Trento logo"
