import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#from torch.nn.utils.rnn import pad_sequence

class Text_dataset(Dataset):
    def __init__(self,
                 df: pd.core.frame.DataFrame,
                 sample : bool,
                 size: int = None):

        self.size = size

        if sample:
          assert size != None, "Size must be specified when sampling"
          # Sample data points
          ids = np.random.randint(0, len(df), size)

          # Build new data frame with sampled data
          self.df = pd.DataFrame([df.iloc[i].values for i in ids])

        else:
          self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x, y = self.df.iloc[idx]
        return x, y

# Needed if we want to do batch training
"""
def custom_collate(input_data):
    xs, lens, ys = list(zip(*input_data))
    max_length = max(lens)
    xs = torch.stack([torch.cat([x, torch.zeros(max_length-x.size(0))]) for x in xs]).to(torch.int)
    lens = torch.tensor(lens)
    ys = torch.stack(ys)
    return xs, lens, ys
"""

def get_dataloaders(training_data, test_data, num_workers=0):
  train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=True)
  test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True)
  return train_dataloader, test_dataloader

