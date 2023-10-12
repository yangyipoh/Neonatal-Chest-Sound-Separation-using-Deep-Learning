import time
import torch
from utils import load_model

model_path = r'C:\Users\ypoh0004\Documents\Model\all1\model_best.pt'
model_config = r'C:\Users\ypoh0004\Documents\Model\all1\model.yaml'

# device = torch.device('cpu')
device = torch.device('cuda')
batch = 1
total_runs = 10
total_time = 0
for _ in range(total_runs):
    input_wav = torch.randn(batch, 1, 40_000, device=device)
    model = load_model(model_path, model_config, device)
    model.eval()

    start_time = time.time()
    filtered_sound = model(input_wav)
    end_time = time.time()
    total_time += end_time - start_time

time_per_epoch = (total_time/total_runs)/batch
print(f'Time completed: {time_per_epoch:.4}')    # in seconds
