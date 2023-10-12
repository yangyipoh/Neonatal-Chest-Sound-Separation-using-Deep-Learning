import torch
from torch import Tensor
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import tqdm
import os
import shutil
import random

def signal_noise_ratio(signal: Tensor, noise: Tensor) -> Tensor:
    assert signal.shape == noise.shape, f"signal and noise are expected to have the same shape, but got {signal.shape} and {noise.shape}."
    eps = torch.finfo(signal.dtype).eps
    snr_value = torch.sqrt(torch.sum(signal**2, dim=-1) + eps) \
                    / torch.sqrt(torch.sum(noise**2, dim=-1) + eps)
    return snr_value


class HLSWav(Dataset):
    '''
        Time series Dataset of the .wav files
    '''
    def __init__(self, 
            base_dir, 
            fold, 
            noise_type,  
            num_sources, 
            train_type:int,     # 0=Train, 1=Val, 2=Test
            crop:int=10,
            preprocess=False, 
            use_conv=True,
            gen_preprocess=False,
            snr_stepsize=5,
            snr_min_l=-10,
            snr_min_n=-10,
        ):
        '''
        input:
            base_dir: directory pointing to Train Sets
            idx: list of index
            noise_type: currently only support 'all'
            crop: return time instance
        '''
        super().__init__()
        self.dir = base_dir
        self.type = noise_type
        self.fold = fold
        self.crop = crop
        self.num_sources = num_sources
        self.preprocess = preprocess
        self.train = train_type
        self.use_conv = use_conv
        self.mix_count = 2 if use_conv else 1
        self.stepsize = 5 if train_type==2 else snr_stepsize
        self.min_l = -10 if train_type==2 else snr_min_l
        self.min_n = -10 if train_type==2 else snr_min_n
        self.conv_length = 5 if train_type==0 else 3
        self.stmv_disconnect = ['stmv22.wav', 'cry44.wav', 'Stmov1_1.wav', 'Stmov1_2.wav', 'Stmov1_3.wav', 'cry41.wav']

        # resolve noise_type
        match noise_type:
            case 'general': noises = ['Stmv', 'Cry']
            case 'resp_support': noises = ['Bubble', 'CPAP']
            case 'all': noises = ['Bubble', 'CPAP', 'Stmv', 'Cry', 'NoNoise']
            case 'all_c': noises = ['Bubble', 'CPAP', 'Cry', 'NoNoise']
            case 'NoNoise': noises = ['NoNoise']
            case 'Cry': noises = ['Cry']
            case _: raise KeyError('Unknown noise sound')

        # save all files as a list
        match train_type:
            case 0:
                self.heart_sounds = sorted(os.listdir(os.path.join(base_dir, f'TrainHeart{fold}')))[:-2]
                self.lung_sounds = sorted(os.listdir(os.path.join(base_dir, f'TrainLung{fold}')))[:-2]
            case 1:
                self.heart_sounds = sorted(os.listdir(os.path.join(base_dir, f'TrainHeart{fold}')))[-2:]
                self.lung_sounds = sorted(os.listdir(os.path.join(base_dir, f'TrainLung{fold}')))[-2:]
            case 2: 
                self.heart_sounds = os.listdir(os.path.join(base_dir, f'TestHeart{fold}'))
                self.lung_sounds = os.listdir(os.path.join(base_dir, f'TestLung{fold}'))
            case _:
                raise KeyError('Unknown train_type')
        
        self.noise_sounds = []
        for noise in noises:
            if noise == 'NoNoise':
                self.noise_sounds.append(('NoNoise', None))
                continue

            match noise:
                case 'Stmv': splice=-4
                case 'Cry':  splice=-6
                case 'Bubble': splice=-1
                case 'CPAP': splice=-2
            
            match train_type:
                case 0: files = sorted(os.listdir(os.path.join(base_dir, f'Train{noise}{fold}')))[:splice]
                case 1: files = sorted(os.listdir(os.path.join(base_dir, f'Train{noise}{fold}')))[splice:]
                case 2: files = os.listdir(os.path.join(base_dir, f'Test{noise}{fold}'))
            self.noise_sounds.extend([(noise, file) for file in files])
        
        # generate all samples in a tmp folder 
        if preprocess and gen_preprocess:
            fold = f'test_{noise_type}'
            shutil.rmtree(os.path.join(base_dir, fold))     # clear folder
            os.mkdir(os.path.join(base_dir, fold))          # make folder
            for idx in tqdm.tqdm(range(len(self))):
                mix_wav, heart_wav, lung_wav, noise_wav = self.generate_mixture(idx)
                torchaudio.save(os.path.join(base_dir, fold, f'{idx}_mix.wav'), mix_wav, 4000)
                torchaudio.save(os.path.join(base_dir, fold, f'{idx}_heart.wav'), heart_wav, 4000)
                torchaudio.save(os.path.join(base_dir, fold, f'{idx}_lung.wav'), lung_wav, 4000)
                torchaudio.save(os.path.join(base_dir, fold, f'{idx}_noise.wav'), noise_wav, 4000)

    def __len__(self):
        total_noise = min(len(self.noise_sounds), 5) if self.train == 0 else len(self.noise_sounds)
        total = len(self.heart_sounds)*len(self.lung_sounds)*total_noise*self.mix_count
        return total*5*5

    def __getitem__(self, idx):
        '''
        output:
            mix:  1, F, M
            ref: S, F, M
        '''
        # decode index
        if self.preprocess:
            fold = f'test_{self.type}'
            mix_wav, _ = torchaudio.load(os.path.join(self.dir, fold, f'{idx}_mix.wav'), normalize=True)
            heart_wav, _ = torchaudio.load(os.path.join(self.dir, fold, f'{idx}_heart.wav'), normalize=True)
            lung_wav, _ = torchaudio.load(os.path.join(self.dir, fold, f'{idx}_lung.wav'), normalize=True)
            noise_wav, _ = torchaudio.load(os.path.join(self.dir, fold, f'{idx}_noise.wav'), normalize=True)
        else:
            mix_wav, heart_wav, lung_wav, noise_wav = self.generate_mixture(idx)

        if self.train == 2:
            _, _, noise, _, _, mix_type = self.decode_idx(idx)
            mix_type = 0 if mix_type=='instantaneous' else 1
            match noise[0]:
                case 'Bubble': noise = 1
                case 'CPAP': noise = 2
                case _: noise = 0
            return mix_wav, torch.cat([heart_wav, lung_wav], dim=0), mix_type, noise

        if self.num_sources == 2:
            return mix_wav, torch.cat([heart_wav, lung_wav], dim=0)
        else:
            return mix_wav, torch.cat([heart_wav, lung_wav, noise_wav], dim=0)
    
    def generate_mixture(self, idx):
        def conv_signal(signal, length):
            a = torch.rand(1, 1, length)
            a = a/torch.norm(a)
            return F.conv1d(signal, a, padding='same')
        
        def mix(signal1, signal2, snr):
            scale = torch.squeeze(signal_noise_ratio(signal1, signal2))
            signal2 = scale*signal2
            return signal1+signal2*10**(snr/20)
            
        def random_crop(signal, crop_len):
            start_idx = random.randrange(0, (10-crop_len)*4_000+1)
            signal = signal[:, start_idx:start_idx+crop_len*4_000]
            return signal
        
        # 'XX.wav', 'XY.wav', ('Cry', 'XZ.wav'), 5, --10, 'instantaneous'
        heart, lung, noise, snr_l, snr_n, mix_type = self.decode_idx(idx)
        conv_length = self.conv_length

        # modifications for training
        if self.train == 0:
            snr_l = random.random()*4*self.stepsize+self.min_l
            snr_n = random.random()*4*self.stepsize+self.min_n
            mix_type = 'convolutive'
            conv_length = random.randrange(1, self.conv_length+1, 2)

        # load audio files
        train_type = 'Test' if self.train == 2 else 'Train'
        heart_wav, _ = torchaudio.load(os.path.join(self.dir, f'{train_type}Heart{self.fold}', heart),normalize=True)
        lung_wav, _ = torchaudio.load(os.path.join(self.dir, f'{train_type}Lung{self.fold}', lung), normalize=True)
        if noise[0] != 'NoNoise':
            noise_wav, _ = torchaudio.load(os.path.join(self.dir, f'{train_type}{noise[0]}{self.fold}', noise[1]), normalize=True)
        else:
            noise_wav = torch.zeros_like(heart_wav)
        
        # handle stmv samples that have size < 40,000
        if noise_wav.shape[-1] < 40_000:
            n = noise_wav.shape[-1]
            tmp = torch.zeros_like(heart_wav)
            insert_point = random.randrange(0, 40_000-n)
            end_point = insert_point+n
            tmp[:, insert_point:end_point] = noise_wav
            noise_wav = tmp
            if noise[1] in self.stmv_disconnect:
                heart_wav[:, insert_point:end_point] = 0
                lung_wav[:, insert_point:end_point] = 0

        if mix_type == 'convolutive':
            heart_wav = conv_signal(heart_wav, conv_length)
            lung_wav = conv_signal(lung_wav, conv_length)

        # random crop
        heart_wav = random_crop(heart_wav, self.crop)
        lung_wav=  random_crop(lung_wav, self.crop)
        noise_wav = random_crop(noise_wav, self.crop)

        # mix heart and lung
        mix_wav = mix(heart_wav, lung_wav, snr_l)
        mix_wav = mix(mix_wav, noise_wav, snr_n)

        # normalise mixture
        mix_wav = mix_wav/torch.max(torch.abs(mix_wav))

        return mix_wav, heart_wav, lung_wav, noise_wav

    def decode_idx(self, idx):
        def decode_onestep(idx, size):
            obj_id = idx%size
            idx = idx//size
            return obj_id, idx
        # 0 = instantaneous, 1 = convolutive (if use_conv)
        mix_encode, idx = decode_onestep(idx, self.mix_count)
        match mix_encode:
            case 0: mix = 'instantaneous'
            case 1: mix = 'convolutive'

        # noise
        noise_encode, idx = decode_onestep(idx, len(self.noise_sounds))
        if self.train: noise = random.choice(self.noise_sounds)
        else:          noise = self.noise_sounds[noise_encode]

        # idx in the list
        lung, idx = decode_onestep(idx, len(self.lung_sounds))
        lung = self.lung_sounds[lung]
        
        # idx in the list
        heart, idx = decode_onestep(idx, len(self.heart_sounds))
        heart = self.heart_sounds[heart]

        # -10, -5, 0, 5, 10
        snr_l, idx = decode_onestep(idx, 5)
        snr_l = snr_l*self.stepsize+self.min_l

        # -10, -5, 0, 5, 10
        snr_n, idx = decode_onestep(idx, 5)
        snr_n = snr_n*self.stepsize+self.min_n

        return heart, lung, noise, snr_l, snr_n, mix
