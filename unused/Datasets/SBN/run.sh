#!/bin/sh

# LAr1ND
python Create_GLB_FluxFile.py

# MicroBooNE
sed -e 's/LAr1ND/MB/g' Create_GLB_FluxFile.py > Create_GLB_FluxFile_MB.py

python Create_GLB_FluxFile_MB.py
rm Create_GLB_FluxFile_MB.py

# ICARUS-T600
sed -e 's/LAr1ND/ICARUS/g' Create_GLB_FluxFile.py > Create_GLB_FluxFile_ICARUS.py 

python Create_GLB_FluxFile_ICARUS.py
rm Create_GLB_FluxFile_ICARUS.py

