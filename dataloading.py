import pandas as pd
import numpy as np
from torch.utils.data import Dataset

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

