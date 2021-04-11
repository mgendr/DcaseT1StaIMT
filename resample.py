import librosa
import scipy.io.wavfile as wav
from scipy.signal import resample 
import numpy as np
import glob
import tqdm
import os
from joblib import Parallel, delayed
import multiprocessing
import argparse



NUM_CORES = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
args = parser.parse_args()

INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path
SR_OUT = 18000 # Hz

def down_sample(i):
    file_name = audio_files[i]
    x, sr = librosa.load(file_name, sr = None, mono = False)
    x = x.T
    N = len(x)
    t = int(N/sr)
    N_resample = int(t*SR_OUT)
    x = resample(x, N_resample)
    name = os.path.basename(file_name)
    wav.write(os.path.join(OUTPUT_PATH,name), SR_OUT, np.int16(x*2**15))

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"{OUTPUT_PATH} folder created.")

audio_files = glob.glob(INPUT_PATH +'*.wav')
print(f'kernel : {NUM_CORES}')
print(len(audio_files))
Parallel(n_jobs=int(NUM_CORES))(delayed(down_sample)(f)
            for f in tqdm.tqdm(range(len(audio_files))))
