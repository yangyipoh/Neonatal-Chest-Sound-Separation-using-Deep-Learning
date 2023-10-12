import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler

import yaml
import os
import shutil
from typing import Tuple, Union, Optional
from hlsdata import HLSWav
from models import MaskNet
from loss_fn import SDR_Loss, LogMSE_Loss, SNR_Loss, SASDR_Loss
from utils import save_model


def get_data(config:dict) -> Tuple[DataLoader, DataLoader]:
    train_data = HLSWav(
        base_dir=config['train_dir'], 
        fold=config['fold'],
        noise_type=config['train_type'], 
        crop=config['hyperparam']['crop_len'], 
        num_sources=config['hyperparam']['model_config']['num_sources'],
        train_type=0,
        snr_stepsize=config['hyperparam']['snr_step'],
        snr_min_l=config['hyperparam']['snr_min_l'],
        snr_min_n=config['hyperparam']['snr_min_n'],
    )
    test_data = HLSWav(
        base_dir=config['train_dir'], 
        fold=config['fold'],
        noise_type=config['train_type'], 
        crop=10,
        num_sources=config['hyperparam']['model_config']['num_sources'],
        train_type=1,
        snr_stepsize=config['hyperparam']['snr_step'],
        snr_min_l=config['hyperparam']['snr_min_l'],
        snr_min_n=config['hyperparam']['snr_min_n'],
    )
    train_dataloader = DataLoader(train_data, batch_size=config['hyperparam']['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=config['hyperparam']['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    return train_dataloader, test_dataloader


def get_model(config:dict) -> MaskNet:
    model = MaskNet(**config['hyperparam']['model_config'])
    if config['train_resume'] is not None:
        model.load_state_dict(torch.load(
            config['train_resume'], 
            map_location=torch.device(0)
        ))
    return model


def get_optimizer(config:dict, model:MaskNet) -> Union[optim.SGD, optim.Adam, optim.AdamW]:
    if config['hyperparam']['optimizer'] == 'SGD':
        return optim.SGD(
            model.parameters(), 
            lr=config['hyperparam']['learning_rate'], 
            momentum=0.9, weight_decay=config['hyperparam']['weight_decay']
        )
    if config['hyperparam']['optimizer'] == 'Adam':
        return optim.Adam(
            model.parameters(), 
            lr=config['hyperparam']['learning_rate'], 
            weight_decay=config['hyperparam']['weight_decay'],
            amsgrad=True,
        )
    if config['hyperparam']['optimizer'] == 'AdamW':
        return optim.AdamW(
            model.parameters(), 
            lr=config['hyperparam']['learning_rate'], 
            weight_decay=config['hyperparam']['weight_decay'],
            amsgrad=True,
        )
    raise Exception('Unknown optimizer')


def get_loss_fn(config:dict):
    if config['hyperparam']['loss'] == 'MSE':
        return nn.MSELoss()
    if config['hyperparam']['loss'] == 'SDR':
        return SDR_Loss(torch.tensor(parse_weight(config['hyperparam']['weights'])))
    if config['hyperparam']['loss'] == 'LogMSE':
        return LogMSE_Loss(torch.tensor(parse_weight(config['hyperparam']['weights'])))
    if config['hyperparam']['loss'] == 'SNR':
        return SNR_Loss(torch.tensor(parse_weight(config['hyperparam']['weights'])))
    if config['hyperparam']['loss'] == 'SASDR':
        return SASDR_Loss(torch.tensor(parse_weight(config['hyperparam']['weights'])))
    raise Exception('Unknown Loss Function')


def parse_weight(weights:str):
    weights_str_lst = weights.split(',')
    return [float(x) for x in weights_str_lst]


def train(
        model:MaskNet, 
        dataloader:DataLoader, 
        loss_fn:Union[nn.MSELoss, SDR_Loss, LogMSE_Loss, SNR_Loss], 
        optimiser:Union[optim.SGD, optim.Adam, optim.AdamW], 
        clip:Optional[float], 
        device:torch.device,
        verbose:bool=False,
    ) -> float:
    model.to(device)
    model.train()

    running_loss = 0
    last_loss = None
    n_print_iter = len(dataloader)//5
    for i, (inpt, target) in enumerate(dataloader):
        # transfer to GPU
        inpt = inpt.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimiser.zero_grad()

        # forward pass
        output = model(inpt)
        loss = loss_fn(output, target)

        # backward pass
        loss.backward()
        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimiser.step()
        running_loss += loss.item()

        # returning results every 1000 iterations
        if (i)%n_print_iter == n_print_iter-1:
            last_loss = running_loss/n_print_iter
            if verbose: print(f'batch {i+1} loss: {last_loss:.6f}')
            running_loss = 0
    return last_loss


def test(
        model:MaskNet, 
        dataloader:DataLoader, 
        loss_fn:Union[nn.MSELoss, SDR_Loss, LogMSE_Loss, SNR_Loss],
        device:torch.device,
    ) -> float:
    model.to(device)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inpt, target in dataloader:
            # transfer to GPU
            inpt = inpt.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # get output
            output = model(inpt)

            # mse
            loss = loss_fn(output, target)
            total_loss += loss.item()
    return total_loss/len(dataloader)


def fit(
        model:MaskNet, 
        train_dataloader:DataLoader, 
        val_dataloader:DataLoader, 
        loss_fn:Union[nn.MSELoss, SDR_Loss, LogMSE_Loss, SNR_Loss], 
        optimiser:Union[optim.SGD, optim.Adam, optim.AdamW], 
        scheduler:lr_scheduler.ReduceLROnPlateau, 
        device:torch.device, 
        config:dict,
        earlystop_patience:int=4,
    ):
    epochs = config['hyperparam']['epochs']
    lowest_loss = float('inf')
    current_lr = optimiser.param_groups[0]['lr']
    for i in range(epochs):
        # train
        train_loss = train(
            model=model, 
            dataloader=train_dataloader, 
            loss_fn=loss_fn, 
            optimiser=optimiser, 
            clip=config['hyperparam']['clip'], 
            device=device,
        )
        
        # test
        val_loss = test(
            model=model, 
            dataloader=val_dataloader, 
            loss_fn=loss_fn, 
            device=device,
        )
        
        # save model
        if val_loss < lowest_loss:
            lowest_loss = val_loss
            save_model(model, os.path.join(config['model_dir'], 'model_best.pt'))
        
        # reduce lr if required
        scheduler.step(val_loss)
        if current_lr != optimiser.param_groups[0]['lr']:
            current_lr = optimiser.param_groups[0]['lr']
            earlystop_patience -= 1
        if earlystop_patience <= 0:
            break
        
        print(f'[{i+1}|{epochs}] Train loss: {train_loss:.6f}, Test loss: {val_loss:.6f}')
    save_model(model, os.path.join(config['model_dir'], 'model_last.pt'))

def main(config:dict):
    # check for CUDA
    assert torch.cuda.is_available(), 'Unable to use CUDA'
    device = torch.device(0)

    # ans = input("Start training? (type no to cancel): ")
    # if ans == 'no' or ans == 'No': return
    
    # model, dataloader, optimizer, loss_fn, scheduler
    model = get_model(config)
    train_dataloader, val_dataloader = get_data(config)
    optimiser = get_optimizer(config, model)
    loss_fn = get_loss_fn(config)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimiser, 
        factor=config['hyperparam']['factor'], 
        patience=config['hyperparam']['patience'],
        verbose=True,
        # threshold=0,
    )
    earlystop = config['hyperparam']['earlystop_patient']
    # scheduler = lr_scheduler.ExponentialLR(
    #     optimiser, gamma=0.96,
    # )

    # clear model folder
    if os.path.exists(config['model_dir']):  shutil.rmtree(config['model_dir'])
    os.makedirs(config['model_dir'])

    # save model configurations
    with open(os.path.join(config['model_dir'], 'model.yaml'), 'w') as f: yaml.dump(config['hyperparam']['model_config'], f)
    
    # train
    print('-------- Start of Training --------')
    fit(model,train_dataloader,val_dataloader,loss_fn,optimiser,scheduler,device,config,earlystop)

    # save training config 
    with open(os.path.join(config['model_dir'], 'config.yaml'), 'w') as f: 
        config.pop('eval')
        yaml.dump(config, f)
    

if __name__ == '__main__':
    # read config file
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    main(config)
