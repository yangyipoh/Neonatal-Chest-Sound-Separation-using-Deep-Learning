import yaml
import argparse
import tqdm
import os
import datetime
import numpy as np
import pandas as pd
import torch
import torchaudio
from utils import load_model, bandpass_filter
from utils.matlab_interface import NMF, OldMethod, MATLABFunction

from torch.utils.data import DataLoader
from hlsdata import HLSWav
from torchmetrics.audio.sdr import (scale_invariant_signal_distortion_ratio, signal_distortion_ratio,)
from fast_bss_eval import (bss_eval_sources, si_sdr)

class HRBRConfig:
    def __init__(self):
        self.nil = [9,11,14,15,17,19,20,24,25,26,28,29,34,35,38,39,40,41,42]
        self.cpap = [1,3,5,12,13,21]
        self.bubble = [16,22,30,31,36,37]
        self.train_idx = self.nil+self.bubble+self.cpap

def get_model(config, device, num_workers:int=3):
    match config['model_type']:
        case 'nn':
            model = load_model(config['model_dir'], config['model_config'], device)
            model.eval()
        case 'NMF': model = NMF('NMF', fold=config['fold'], num_workers=num_workers)
        case 'NMCF': model = NMF('NMCF', fold=config['fold'], num_workers=num_workers)
        case _: model = OldMethod(config['model_type'], num_workers=num_workers)
    return model

def model_evaluate(model, model_type, mix, noise, device, bandpass=False):
    with torch.no_grad():
        if model_type == 'nn':
            if bandpass:
                mix = bandpass_filter(mix, 0.0125, 0.25, 51)
            output = model(mix)
        elif model_type == 'NMF' or model_type == 'NMCF':
            output = model(mix.cpu(), noise)    # for NMF and NMCF
            output = output.to(device)
        else:
            output = model(mix.cpu())
            output = output.to(device)
        output = output[:, 0:2, :]  # (1, 2, length)
    return output

def hrbr_preprocess(fs, seg_size, config, hrbr_config:HRBRConfig, mlab:MATLABFunction):
    # load example dataset
    path = os.path.join(config['eval_dir_sync'], 'ExamplesDemo.xlsx')
    df_main = pd.read_excel(path, sheet_name=0)

    df_out = {
        'Filename': [],
        'Sample': [],
        'Resp Support': [],
        'Truth HR': [],
        'Truth BR': [],
        'Baseline HR': [],
        'Baseline BR': [],
    }

    pbar = tqdm.tqdm(total=len(df_main['Example']), desc='Preprocessing')
    for sample in range(len(df_main['Example'])):
        if (sample+1) not in hrbr_config.train_idx:
            pbar.update(1)
            continue

        # open wavefile
        path = os.path.join(config['eval_dir_sync'], 'Data', 'Audio Files', f'{df_main["Example"][sample]}.wav')
        waveform, fs_wav = torchaudio.load(path, normalize=True)
        waveform = torchaudio.functional.resample(
            waveform=waveform,
            orig_freq=fs_wav,
            new_freq=fs,
            lowpass_filter_width=6,
            rolloff=0.8,
        )

        # read sync file
        path = os.path.join(config['eval_dir_sync'], 'Data', 'Sync Data', f"{df_main['Example'][sample]}.txt")
        df_sync = pd.read_csv(path, sep='\t')

        # get when the segment starts
        start_time:datetime.time = df_main['Time Taken'][sample]

        # sample data
        resp_support = 0 if df_main['Respiratory Support'][sample] == 'Nil' else 1

        for segment in range(5, waveform.shape[1]//fs-seg_size, 10):
            # time_taken + segment offset
            time_taken = datetime.datetime(100, 1, 1, start_time.hour, start_time.minute, start_time.second)
            time_taken += datetime.timedelta(seconds=segment)

            # create mask, get vital for time segment, convert it to float
            time_segment = (str(time_taken.time()) <= df_sync['TIME']) & (df_sync['TIME'] < str((time_taken+datetime.timedelta(seconds=seg_size)).time()))
            sample_vital = df_sync.loc[time_segment]
            sample_vital = sample_vital.mask(sample_vital['HR'] == '***')   # deal with illegal character
            sample_vital = sample_vital.mask(sample_vital['RESP'] == '***')
            sample_vital = sample_vital.mask(sample_vital['HR'] == ' ')
            sample_vital = sample_vital.mask(sample_vital['RESP'] == ' ')
            sample_vital = sample_vital.mask(sample_vital['HR'] == '   ')
            sample_vital = sample_vital.mask(sample_vital['RESP'] == '   ')
            sample_vital = sample_vital.astype({'HR':'float', 'RESP':'float'})

            # filename, HR, BR
            filename = f'{sample+1}_{segment}.wav'
            hr_truth = sample_vital['HR'].mean()
            br_truth = sample_vital['RESP'].mean()

            if np.isnan(hr_truth) or np.isnan(br_truth):
                continue

            # export
            df_out['Filename'].append(filename)
            df_out['Sample'].append(sample+1)
            df_out['Resp Support'].append(resp_support)
            df_out['Truth HR'].append(hr_truth)
            df_out['Truth BR'].append(br_truth)
            
            waveform_seg = waveform[:1, segment*fs:(segment+seg_size)*fs]

            # baseline HR and BR
            waveform_seg_numpy = np.squeeze(waveform_seg.detach().numpy())
            waveform_seg_numpy = np.expand_dims(waveform_seg_numpy, axis=0)
            hr_base = mlab.get_hr(waveform_seg_numpy, fs)
            br_base = mlab.get_br(waveform_seg_numpy, fs)
            df_out['Baseline HR'].append(hr_base)
            df_out['Baseline BR'].append(br_base)

            path = os.path.join(config['eval_dir_sync'], 'Data', 'Audio Files', 'Sync Segmented', filename)
            torchaudio.save(path, waveform_seg, fs)
        
        pbar.update(1)

    path = os.path.join(config['eval_dir_sync'], 'Data', 'Audio Files', 'Sync Segmented', 'data.csv')
    df = pd.DataFrame(data=df_out)
    df.to_csv(path, index=False)
    return df

def eval_hrbr(config):
    def parse_noise(sample:str, hrbr_config:HRBRConfig):
        sample = sample.split('_')[0]
        if sample in hrbr_config.bubble: return 'Bubble'
        elif sample in hrbr_config.cpap: return 'CPAP'
        return 'none'

    # variables
    hrbr_config = HRBRConfig()
    device = torch.device(config['device'])
    seg_size = 10
    fs = 4000

    # get preprocessed data (generate or read)
    mlab = MATLABFunction(num_workers=1)
    df = hrbr_preprocess(fs, seg_size, config, hrbr_config, mlab) if config['preprocess'] else pd.read_csv(os.path.join(config['eval_dir_sync'], 'Data', 'Audio Files', 'Sync Segmented', 'data.csv'))
    
    # calculate metric
    model = get_model(config, device, num_workers=1)

    predict_hr = []
    predict_br = []
    for sample in tqdm.tqdm(range(len(df['Filename'])), desc='Processing'):
        # read audio file
        path = os.path.join(config['eval_dir_sync'], 'Data', 'Audio Files', 'Sync Segmented', df['Filename'][sample])
        mix, fs = torchaudio.load(path, normalize=True)
        mix = torch.unsqueeze(mix, dim=0)
        mix = mix.to(device)

        # pass mix to model and extract heart and lung
        noise = parse_noise(df['Filename'][sample], hrbr_config)
        filtered = model_evaluate(model, config['model_type'], mix, [noise], device, bandpass=True)
        heart = filtered[0, 0, :].cpu().detach().numpy()
        lung = filtered[0, 1, :].cpu().detach().numpy()
        heart /= np.max(np.abs(heart))
        lung /= np.max(np.abs(lung))
        heart = np.expand_dims(heart, axis=0)
        lung = np.expand_dims(lung, axis=0)

        # predicted HR and BR
        hr = mlab.get_hr(heart, fs)
        br = mlab.get_br(lung, fs)
        predict_hr.append(hr)
        predict_br.append(br)

    df['Predict HR'] = predict_hr
    df['Predict BR'] = predict_br


    # postprocessing (separated just in case postprocessing is needed)
    hr_improve = []
    br_improve = []
    for _, (_, _, _, hr_truth, br_truth, hr_base, br_base, hr, br) in df.iterrows():
        hr_improve.append(abs(hr_truth-hr_base) - abs(hr_truth-hr))
        br_improve.append(abs(br_truth-br_base) - abs(br_truth-br))
    df['HR improvement'] = hr_improve
    df['BR improvement'] = br_improve

    # export results
    df.to_csv(config['eval_file_hrbr'], index=False)

def signal_x_ratio(est, ref, use_cg_iter=10):
    sdr = signal_distortion_ratio(preds=est, target=ref, use_cg_iter=use_cg_iter)
    si_sdr = scale_invariant_signal_distortion_ratio(preds=est, target=ref)
    return sdr, si_sdr

def signal_x_ratio_eval(est, ref, use_cg_iter=10):
    # est, ref should have the shape of (..., n_channels, n_samples)
    res_sdr, res_sir, _ = bss_eval_sources(ref, est, compute_permutation=False, use_cg_iter=use_cg_iter)
    res_si_sdr = si_sdr(ref, est, return_perm=False)
    return res_sdr, res_sir, res_si_sdr     # return shape (..., n_channels)

def append_sxr_metrics(lst:list, before, after, mix_type):
    improve = [after[i] - before[i] for i in range(len(before))]
    batch = before[0].shape[0]
    for idx in range(batch):
        lst.append(pd.DataFrame({
            'mixture type':mix_type[idx].item(),
            'mixture-heart SDR':before[0][idx, 0].item(),
            'mixture-heart SI-SDR':before[2][idx, 0].item(),
            'mixture-heart SIR':before[1][idx, 0].item(),
            'mixture-lung SDR':before[0][idx, 1].item(),
            'mixture-lung SI-SDR':before[2][idx, 1].item(),
            'mixture-lung SIR':before[1][idx, 1].item(),
            'filtered-heart SDR':after[0][idx, 0].item(),
            'filtered-heart SI-SDR':after[2][idx, 0].item(),
            'filtered-heart SIR':after[1][idx, 0].item(),
            'filtered-lung SDR':after[0][idx, 1].item(),
            'filtered-lung SI-SDR':after[2][idx, 1].item(),
            'filtered-lung SIR':after[1][idx, 1].item(),
            'SDR improvement heart':improve[0][idx, 0].item(),
            'SI-SDR improvement heart':improve[2][idx, 0].item(),
            'SIR improvement heart':improve[1][idx, 0].item(),
            'SDR improvement lung':improve[0][idx, 1].item(),
            'SI-SDR improvement lung':improve[2][idx, 1].item(),
            'SIR improvement lung':improve[1][idx, 1].item(),
        }, index=[0]))

def eval_sxr(config):
    def decode_noise(noise_single):
        match noise_single:
            case 0: return 'none'
            case 1: return 'Bubble'
            case 2: return 'CPAP'

    # load model
    device = torch.device(config['device'])
    model = get_model(config, device)

    # dataset
    test_data = HLSWav(config['eval_dir'], config['fold'],config['noise_type'],2,2, preprocess=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # iterate through dataset
    lst_result = []
    for mix, target, mix_type, noise in tqdm.tqdm(test_dataloader, desc=f'{config["noise_type"]}'):
        mix = mix.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # base case (B, 2)
        with torch.no_grad():
            sdr_before, sir_before, sisdr_before = signal_x_ratio_eval(mix.repeat(1, 2, 1), target)
        
        # pass data to model
        noise = [decode_noise(elem.item()) for elem in noise]   # convert noise_idx to noise
        output = model_evaluate(model, config['model_type'], mix, noise, device)

        # evaulate (B, 2)
        with torch.no_grad():
            sdr_after, sir_after, sisdr_after = signal_x_ratio_eval(output, target)

        # compute the improvement
        append_sxr_metrics(lst_result, (sdr_before, sir_before, sisdr_before), (sdr_after, sir_after, sisdr_after), mix_type)
    
    df = pd.concat(lst_result, ignore_index=True)
    df.to_csv(config['eval_file_sxr'], index=False)

def eval_sxr_all(config):
    for noise in ['NoNoise', 'resp_support', 'general']:
        config['noise_type'] = noise
        config['eval_file_sxr'] = f'result/{noise}.csv'
        eval_sxr(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Code')
    parser.add_argument('-opt', '--options', type=str, default='all', help='Evaluation Options (all, sxr_all, sxr, hrbr)')
    args = parser.parse_args()
    option = args.options

    # get dictionary
    with open('config_eval.yaml', 'r') as f:
        config = yaml.safe_load(f)

    match option:
        case 'all':
            eval_sxr_all(config)
            eval_hrbr(config)
        case 'sxr_all':
            eval_sxr_all(config)
        case 'sxr':
            eval_sxr(config)
        case 'hrbr':
            eval_hrbr(config)
        case _: raise KeyError(f'-opt parameter only accepts (all, sxr_all, sxr, hrbr), received {option}')
    print("----Evaluation Complete----")
