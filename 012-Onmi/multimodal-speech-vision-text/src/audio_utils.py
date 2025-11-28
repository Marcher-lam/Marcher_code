import torchaudio

def load_audio(path,target_sr=16000):
    waveform,orig_sr = torchaudio.load(path)
    if orig_sr != target_sr:
        waveform = torchaudio.functions.resample(waveform,orig_sr,target_sr)
    return waveform

def compute_mel_spec(waveform,sr=16000,n_mels=80):
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_mels=n_mels)(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    return mel_db
