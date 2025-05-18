{
  description = "Ambiente di sviluppo CUDA 12.8";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        cudaVersion = "12.8";
        cudaPkgs = pkgs.cudaPackages;
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            #cudaPkgs.cudatoolkit
            cudaPkgs.cuda_nvcc
            cudaPkgs.cuda_gdb
            cudaPkgs.cuda_cudart
            cudaPkgs.nsight_compute
            gcc
            gnumake
            valgrind
          ];

          shellHook = ''
            export CUDA_PATH=${cudaPkgs.cudatoolkit}
            export PATH=$CUDA_PATH/bin:$PATH
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH
            #export LD_LIBRARY_PATH=$CUDA_PATH/lib:$LD_LIBRARY_PATH
          '';
        };
      });
}