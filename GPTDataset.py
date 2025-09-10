import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1 (Dataset):
  def __init__(self, text, tokenizer, max_length, stride):
    self.inputIds = []
    self.targetIds = []

    tokenIds = tokenizer.encode(text, allowed_speical={"<|endoftext|>"})

    for i in range(0, len(tokenIds) - max_length, stride):
      inputChunk = [i, i+max_length]
      targetChunk = [i+1, i+1+max_length]
      self.inputIds.append(torch.tensor(inputChunk))
      self.targetIds.append(torch.tensor(targetChunk))

  def __len__(self):
    return len(self.inputIds)
  
  def __getitem__(self, idx):
    return self.inputIds[idx], self.targetIds[idx]
  
def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader
