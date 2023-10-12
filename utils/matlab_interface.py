import matlab
import matlab.engine
import torch
from torch import Tensor
import numpy as np
from typing import Optional


class MATLABFunction:
    def __init__(self, num_workers:int=1) -> None:
        print('Starting MATLAB')
        self.mlab = [self.get_engine() for _ in range(num_workers)]
        print('Started MATLAB')
        self.num_workers=num_workers
    
    def get_engine(self):
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath('heart_sound_analysis'), nargout=0)
        eng.addpath('utils/matlab_code', nargout=0)
        return eng

    def quit(self) -> None:
        for eng in self.mlab:
            eng.quit()
        self.mlab = None
    
    def tensor2array(self, arr_numpy:Tensor):
        return matlab.double(arr_numpy.tolist())

    def array2tensor(self, arr_double):
        arr = np.asarray(arr_double.tomemoryview())
        return torch.tensor(arr, dtype=torch.float32)
    
    def get_hr(self, heart_wav, fs:int=4000):
        heart_wav_arr = self.tensor2array(heart_wav)
        hr = self.mlab[0].get_hr(heart_wav_arr, float(fs))
        return hr
    
    def get_br(self, lung_wav, fs:int=4000):
        lung_wav_arr = self.tensor2array(lung_wav)
        br = self.mlab[0].get_br(lung_wav_arr, float(fs))
        return br
    

class NMF(MATLABFunction):
    def __init__(self, nmf_type:str, pretrain:bool=False, path:Optional[str]=None, fold:int=1, num_workers=2) -> None:
        super().__init__(num_workers=num_workers)
        assert nmf_type == 'NMF' or nmf_type == 'NMCF', 'Unknown NMF type (NMF or NMCF)'
        self.type = nmf_type        # NMF or NMCF
        if pretrain:
            assert path is not None, 'Argument path is not given when pretraining is selected' 
            self.pretrain_nmf(path, fold)    

    def pretrain_nmf(self, path, fold):
        for noise in ['none', 'CPAP', 'Bubble']:
            print(f'------ Pretraining {noise} ------')
            self.mlab[0].pretrain_nmf(noise, path, fold, nargout=0)
        print('------ Pretraining complete ------')

    def batch_nmf(self, mix:Tensor, noise_lst:list[str], method:str='NMF', fs:int=4000):
        futures = []
        for i in range(len(noise_lst)):
            mix_arr = self.tensor2array(mix[i, :])
            noise = noise_lst[i]
            match method:
                case 'NMF': future = self.mlab[i].get_nmf(mix_arr, fs, noise, background=True)
                case 'NMCF': future = self.mlab[i].get_nmcf(mix_arr, fs, noise, background=True)
                case _: raise KeyError(f'Method in batch_nmf must be NMF or NMCF, received {method}')
            futures.append(future)
        return [self.array2tensor(future.result()) for future in futures]

    def __call__(self, x:Tensor, noise:list) -> Tensor:
        # Input shape -> (B, 1, T)
        # Output shape -> (B, S, T)
        # noise must be 'none', 'Bubble', 'CPAP'
        output = []
        for i in range(0, x.shape[0], self.num_workers):
            x_minibatch = x[i:i+self.num_workers, :, :]
            noise_minibatch = noise[i:i+self.num_workers]
            output += self.batch_nmf(x_minibatch, noise_minibatch, self.type)
        return torch.stack(output)
    

class OldMethod(MATLABFunction):
    def __init__(self, method:str, num_workers:int=2) -> None:
        super().__init__(num_workers=num_workers)
        self.method = method
        self.check_method()

    def batch_traditional_method(self, mix:Tensor, method:str):
        futures = []
        for i in range(mix.shape[0]):
            mix_arr = self.tensor2array(mix[i, :])
            future = self.mlab[i].get_traditional_method(mix_arr, method, background=True)
            futures.append(future)
        return [self.array2tensor(future.result()) for future in futures]

    def __call__(self, x:Tensor) -> Tensor:
        # Input shape -> (B, 1, T)
        # Output shape -> (B, S, T)
        output = []
        for i in range(0, x.shape[0], self.num_workers):
            x_minibatch = x[i:i+self.num_workers, :, :]
            output += self.batch_traditional_method(x_minibatch, self.method)
        return torch.stack(output)
    
    def check_method(self):
        valid_method = ['ale', 'rls', 'wtst', 'fi', 'mf', 'ssa', 'sf', 'aft', 'emd', 'wssa', 'nmfc1', 'nmfc2']
        assert self.method in valid_method, f'Invalid method provided to OldMethod. Provided method was {self.method}'
    