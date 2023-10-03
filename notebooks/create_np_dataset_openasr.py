import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import glob
import sys
sys.path.append("../")
from bengali_asr.dataset.encoder_decoder_dataset import load_audio
# Dummy function to simulate your actual processing function
from tqdm import tqdm

def save_processed_data(output_dir, file_name, data):
    output_file_path = os.path.join(output_dir, f"{file_name}.npy")
    np.save(output_file_path, data)

def process_and_save(file_path, output_dir):
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Process the mp3 file
    processed_data = load_audio(file_path,16000)
    
    # Save the processed data
    save_processed_data(output_dir, file_name, processed_data)

if __name__ == "__main__":
    # List of mp3 file paths
    mp3_file_paths= glob.glob("/app/openASR/dataset/*")
    
    # Output directory
    output_dir = "/app/dataset/train_numpy_16k"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Number of workers (adjust based on your CPU)
    num_workers = 8

    # Process mp3 files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_and_save, mp3_file_paths, [output_dir]*len(mp3_file_paths)), total=len(mp3_file_paths)))
