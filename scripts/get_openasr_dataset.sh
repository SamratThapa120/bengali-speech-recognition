wget https://us.openslr.org/resources/53/utt_spk_text.tsv
wget https://us.openslr.org/resources/53/asr_bengali_a.zip
wget https://us.openslr.org/resources/53/asr_bengali_b.zip
wget https://us.openslr.org/resources/53/asr_bengali_c.zip
wget https://us.openslr.org/resources/53/asr_bengali_d.zip
wget https://us.openslr.org/resources/53/asr_bengali_e.zip
wget https://us.openslr.org/resources/53/asr_bengali_f.zip
wget https://us.openslr.org/resources/53/asr_bengali_0.zip
wget https://us.openslr.org/resources/53/asr_bengali_1.zip
wget https://us.openslr.org/resources/53/asr_bengali_2.zip
wget https://us.openslr.org/resources/53/asr_bengali_3.zip
wget https://us.openslr.org/resources/53/asr_bengali_4.zip
wget https://us.openslr.org/resources/53/asr_bengali_5.zip
wget https://us.openslr.org/resources/53/asr_bengali_6.zip
wget https://us.openslr.org/resources/53/asr_bengali_7.zip
wget https://us.openslr.org/resources/53/asr_bengali_8.zip
wget https://us.openslr.org/resources/53/asr_bengali_9.zip

#!/bin/bash

# List of zip files to process
files=("asr_bengali_0.zip" "asr_bengali_1.zip" "asr_bengali_2.zip" "asr_bengali_3.zip" "asr_bengali_4.zip" "asr_bengali_5.zip" "asr_bengali_6.zip" "asr_bengali_7.zip" "asr_bengali_8.zip" "asr_bengali_9.zip" "asr_bengali_a.zip" "asr_bengali_b.zip" "asr_bengali_c.zip" "asr_bengali_d.zip" "asr_bengali_e.zip" "asr_bengali_f.zip")

# Loop through each file in the list
for file in "${files[@]}"; do
    # Extract the base name of the file without the .zip extension
    base_name=$(basename "$file" .zip)
    
    # Unzip the file
    unzip "$file"
    
    # Move the .flac files to the dataset directory
    mv asr_bengali/data/*/*.flac ./dataset/
    
    # Remove the extracted directory
    rm -rf asr_bengali
done
