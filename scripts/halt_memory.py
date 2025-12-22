import torch
import time
import multiprocessing as mp


device = 'cuda'

def run(rank: int):
    torch.cuda.set_device(rank)

    while True:
        tensor = torch.rand(10, 1024, 1024, 256, device=device)
        t = tensor + tensor
        time.sleep(0.05)
     
        
process_list = list()
for i in [2,]:
    p = mp.Process(target=run, args=(i,))
    process_list.append(p)
    p.start()

for p in process_list:
    p.join()
