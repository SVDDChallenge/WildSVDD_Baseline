import numpy as np
import soundfile as sf
import torch, os
from torch import Tensor
from torch.utils.data import Dataset
import librosa


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_WildSVDD(Dataset):
    def __init__(self, base_dir="/data2/neil/SVDD_challenge/WildSVDD/WildSVDD_Data_Sep2024_Processed", subfolder="train", is_mixture=False, target_sr=16000):
        """
        base_dir contains subfolders for each set: train, test_A, test_B
        each subfolder should contain mixture/ and vocals/ folders
        """
        self.base_dir = base_dir
        self.is_mixture = is_mixture
        self.target_sr = target_sr
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        
        # get file list
        self.file_list = []
        if self.is_mixture:
            self.target_path = os.path.join(self.base_dir, subfolder, "mixture")
        else:
            self.target_path = os.path.join(self.base_dir, subfolder, "vocals")
            
        print(self.target_path)
        
        assert os.path.exists(self.target_path), f"{self.target_path} does not exist!"
        
        for file in os.listdir(self.target_path):
            if file.endswith(".flac"):
                self.file_list.append(file[:-5])

        self.label_dict = {"bonafide": 0, "deepfake": 1}
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        key = self.file_list[index]
        key_split = key.split("_")
        if len(key_split) == 3:
            label, title, clip_idx = key_split
        elif len(key_split) > 3:
            label = key_split[0]
            clip_idx = key_split[-1]
            title = "_".join(key_split[1:-1])
        file_path = os.path.join(self.target_path, key + ".flac")
        # X, _ = sf.read(file_path, samplerate=self.target_sr)
        try:
            X, _ = librosa.load(file_path, sr=self.target_sr, mono=False)
        except:
            return self.__getitem__(np.random.randint(len(self.file_list)))
        if X.shape[0] > 1:
            # if not mono, take random channel
            channel_id = np.random.randint(X.shape[0])
            X = X[channel_id]
            # X = np.expand_dims(X, axis=-1)
        X_pad = pad_random(X, self.cut)
        if np.max(np.abs(X_pad)) == 0:
            print(file_path)
            return self.__getitem__(np.random.randint(len(self.file_list)))
        X_pad = X_pad / np.max(np.abs(X_pad))
        x_inp = Tensor(X_pad)
        
        y = self.label_dict[label]

        return x_inp, y, title, clip_idx
    

if __name__ == "__main__":
    # test the Dataset_WildSVDD
    dataset = Dataset_WildSVDD()
    print(len(dataset))
    x, y, title, clip_idx = dataset[1230]
    print(x.shape, y, title, clip_idx)



