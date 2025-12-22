import torch


class VirtualLoader:
    """
    生成随机数据用于debug
    """
    def __init__(self, channels: int, size: list[int], total=1, return_num=1):
        self.channels = channels
        self.size = size
        self.total = total
        self.current = 0
        self.return_num = return_num

        return
    
    def __iter__(self):
        return self
    
    def __getitem__(self, idx):
        return [torch.rand(self.channels, *self.size) for _ in range(self.return_num)]

    def __next__(self):
        if self.current < self.total:
            self.current += 1
            return [torch.rand(self.channels, *self.size) for _ in range(self.return_num)]
        else:
            self.current = 0
            raise StopIteration
        
    def __len__(self):
        return self.total
