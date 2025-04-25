
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  name = "cuda-dev-shell";

  buildInputs = with pkgs; [
    cudaPackages.cudatoolkit
    cudaPackages.cuda_nvcc
    cudaPackages.cuda_cudart
    gcc
    gnumake
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$CUDA_PATH/lib:$CUDA_PATH/lib64:$LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=$CUDA_PATH/include:$CPLUS_INCLUDE_PATH
  '';
}  
