from torch.utils.data import Dataset
import torch
import os 

class WaveformsDataset(Dataset):
  
    def __init__(self, base_dir, metadata):
        self.metadata = metadata
        self.base_dir = base_dir
  
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx:int):
        temp = self.metadata[idx]
        x = self.open_window(self.base_dir, temp[0], temp[1])
        y = torch.tensor(self.metadata[idx][2])
        return x, y

    def open_tensor(self, path_to_tensor: str):
        with open(path_to_tensor, "rb") as f:
            t = torch.load(f)
            return t

    def open_window(self, base_dir, start, end):
        dut_folder = start.split('_')[0] + "_" + start.split('_')[1]
        dut_folder_full_path = os.path.join(base_dir, dut_folder)

        start_cycle = int(start.split('_')[2])
        end_cycle = int(end.split('_')[2])
        t = list()

        for i in range(start_cycle, end_cycle+1):
            temp = self.open_tensor(os.path.join(dut_folder_full_path, f"{dut_folder}_{i}.pt"))
            temp = temp[:2, 10:]
            t.append(temp)
        x = torch.stack(t)
        return(x)
