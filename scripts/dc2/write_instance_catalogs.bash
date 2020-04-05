#!/bin/bash

# ObsHistId
visit_id=$1

# Point to sims_GCRCatSimInterface using env variable set by lsst stack
gcr_catsim_dir=$SIMS_GCRCATSIMINTERFACE_DIR

# Output folder for instance catalog files
# out_dir=$2
out_dir=$2/"$(printf "%08d" $visit_id)"

# Location of lensed agn truth catalog
agn_truth_cat=$3

# Location of lensed sne truth catalog
sne_truth_cat=$4

# Folder inside catalog output folder to store SED files for lensed SNe
sne_sed_dir=$5

python $gcr_catsim_dir/bin.src/generateInstCat.py \
    --config_file $gcr_catsim_dir/workspace/run2.1/test_agn_config_file_2.1.ddf.json \
    --fov 0.05 --ids $visit_id --agn_threads 2 --out_dir $out_dir

python create_agn_ic.py \
    --obs_db /global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db \
    --obs_id $visit_id --agn_truth_cat $agn_truth_cat --file_out $out_dir/lensed_agn_$visit_id.txt

python create_sne_ic.py \
    --obs_db /global/projecta/projectdirs/lsst/groups/SSim/DC2/minion_1016_desc_dithered_v4.db \
    --obs_id $visit_id --sne_truth_cat $sne_truth_cat --output_dir $out_dir \
    --cat_file_name lensed_sne_$visit_id.txt --sed_folder $sne_sed_dir
