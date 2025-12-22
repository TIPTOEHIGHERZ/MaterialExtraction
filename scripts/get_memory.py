import torch
import time
import multiprocessing as mp


device = 'cuda'

def run(rank: int):
    torch.cuda.set_device(rank)
    
    tensor_list = list()
    while True:
        try:
            tensor = torch.rand(1, 4, 1024, 256, device=device)
            tensor_list.append(tensor)
        except Exception as e:
            time.sleep(5.)
            continue
            
        
process_list = list()
for i in range(3, 5):
    p = mp.Process(target=run, args=(i,))
    process_list.append(p)
    p.start()

for p in process_list:
    p.join()
