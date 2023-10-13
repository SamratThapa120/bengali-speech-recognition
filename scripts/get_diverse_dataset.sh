# !/bin/bash

# Define the datasets and their respective paths
declare -A datasets=(
    ["kathbath"]="https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/kathbath.zip"
    ["kathbath_noisy"]="https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/kathbath_noisy.zip"
    ["commonvoice"]="https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/commonvoice.zip"
    ["fleurs"]="https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/fleurs.zip"
    ["indictts"]="https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/indictts.zip"
    ["mucs"]="https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/mucs.zip"
    ["gramvaani"]="https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/gramvaani.zip"
)

# Download and process the datasets
for dataset in "${!datasets[@]}"; do
    wget -P ./dataset "${datasets[$dataset]}"
    unzip "./dataset/$dataset.zip" "$dataset/bengali/*" -d "./dataset"
    unzip "./dataset/$dataset.zip" "$dataset/*/bn/*" -d "./dataset"

    rm "./dataset/$dataset.zip"

    wget -P ./dataset_bmark "${datasets[$dataset]/vistaar/vistaar_benchmarks}"
    unzip "./dataset_bmark/$dataset.zip" "$dataset/bengali/*" -d "./dataset_bmark"
    rm "./dataset_bmark/$dataset.zip"
done


wget -P ./dataset https://objectstore.e2enetworks.net/indic-asr-public/indicwhisper/vistaar/commonvoice.zip
unzip ./dataset/commonvoice.zip -d "./dataset"