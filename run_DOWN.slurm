#!/bin/bash

# settings
#SBATCH --chdir ./
#SBATCH --output ./log/%j.log
#SBATCH --error ./error/%j.err
#SBATCH --mail-user arturo.sanchez@dicea.unipd.it
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 100G
#SBATCH --time 35-00:00:00

# Obtener el nombre del producto desde el argumento
product="$1"
short_product=$(echo "$product" | cut -c1-2 | tr '[:lower:]' '[:upper:]')
scontrol update JobID=$SLURM_JOB_ID JobName="D_${short_product}"

cd ~/downpy/python/

# For 3h from 2002 to 2023
# IMERG3h, CMORPH, ERA53h, GSMaP, MSWEP

if [[ "$product" == "PERSIANN" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr PERSIANN -tr 1dy -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "SM2RAIN" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr SM2RAIN -tr 1dy -ys 2007 -ye 2023 "${@:2}"
elif [[ "$product" == "MSWEP" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr MSWEP -tr 3h -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "CMORPH" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr CMORPH -tr 3h -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "ERA51dy" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr ERA5 -tr 1dy -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "ERA53h" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr ERA5 -tr 3h -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "IMERG1dy" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr IMERG -tr 1dy -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "IMERG3h" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr IMERG -tr 3h -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "CHIRPS" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr CHIRPS -tr 1dy -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "GSMaP3h" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr GSMaP -tr 3h -ys 2002 -ye 2023 "${@:2}"
elif [[ "$product" == "GSMaP1dy" ]]; then
    srun ~/.conda/envs/AXE/bin/python 1_DOWN_v1.py -pr GSMaP -tr 1dy -ys 2002 -ye 2023 "${@:2}"
else
    echo "Error: $product not found."
    echo "Use IMERG3h, IMERG1dy, MSWEP, CMORPH, ERA53h, ERA51dy, PERSIANN, GSMaP3h, GSMaP1dy, SM2RAIN or CHIRPS."
#    exit 1
fi

echo "$error_file"
error_file="error/${SLURM_JOB_ID}.err"
find error/ -type f -name "*.err" -size 0 -delete
#if [[ -f "$error_file" && ! -s "$error_file" ]]; then
#    rm "$error_file"
#fi

