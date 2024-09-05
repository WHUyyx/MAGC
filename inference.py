
import os

# CUDA_VISIBLE_DEVICES=5 python inference.py  \
# --ckpt magc_ckpts/ckpts_stage2/v54_step=129999-lpips=0.3981.ckpt    \
# --config configs/model/cldm.yaml    \
# --input_path ../dataset/Synthetic-v18-45k/test_4500    \
# --steps 50    \
# --batchsize 30    \
# --output_root metrics_4500_magc    \
# --device cuda    

from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.cldm import ControlLDM
from utils.common import instantiate_from_config, load_state_dict

from pathlib import Path
from torchvision import transforms
import json
from collections import defaultdict

from cal_metrics.iqa import single_iqa, get_fid_from_path
single_iqa = single_iqa()



def parse_args() -> Namespace:
    parser = ArgumentParser()
    # TODO: add help info for these options
    parser.add_argument("--ckpt", type=str,default='magc_ckpts/ckpts_stage2/v54_step=129999-lpips=0.3981.ckpt', help="full checkpoint path")
    parser.add_argument("--config", type=str,default='configs/model/cldm.yaml', help="model config path")

    parser.add_argument("--input_path", type=str, default='../dataset/Synthetic-v18-45k/test_4500')
    parser.add_argument("--steps", default = 50, type=int)
    parser.add_argument("--batchsize", default = 30, type=int)

    # latent image guidance
    parser.add_argument("--output_root", type=str, default='metrics_4500_magc')
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    return parser.parse_args()

def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device


def cal_metrics_and_save(samples, filenames, save_path, refs):
    img_gt = samples['img_gt']
    img_rec = samples['samples']

    img_num = len(filenames)
    metrics_batch_total = defaultdict(float)

    # calculating single image
    for i in range(img_num):
        metrics_single = {}
        img_gt_single = img_gt[i,:,:,:].unsqueeze(0)
        img_rec_single = img_rec[i,:,:,:].clamp_(0, 1).unsqueeze(0)
        metrics = single_iqa.cal_metrics(img_gt_single, img_rec_single)
        metrics_single['bpp'] = samples['bpp_list'][i]

        for k in metrics:
            metrics_single[k] = metrics[k]

        for k, v in metrics_single.items():
            metrics_batch_total[k] += v

        _filename = filenames[i].split('.')[0]

        save_path_img = os.path.join(save_path, f'{_filename}.png')
        save_path_metrics = os.path.join(save_path, f'{_filename}.json')


        img_gt_single = transforms.ToPILImage()(img_gt_single.squeeze(0).cpu())
        img_rec_single = transforms.ToPILImage()(img_rec_single.squeeze(0).cpu())
        img_ref_single = refs[i]
        result_img = Image.new('RGB', (img_gt_single.width + img_rec_single.width + img_ref_single.width + 20 , img_gt_single.height))
        result_img.paste(img_gt_single, (0, 0))
        result_img.paste(img_rec_single, (img_gt_single.width + 10, 0))
        result_img.paste(img_ref_single, (img_gt_single.width + img_rec_single.width + 20, 0))
        result_img.save(save_path_img)

        with Path(save_path_metrics).open("wb") as f:
            output = {
                "results": metrics_single,
            }
            f.write(json.dumps(output, indent=2).encode())


    return metrics_batch_total




@torch.no_grad()
def inference_batch(
    model: ControlLDM,
    batch: dict, 
    recon_path: str,
    steps: int,
    ):

    img_tensor = torch.tensor(np.stack(batch['imgs']) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    ref_tensor = torch.tensor(np.stack(batch['refs']) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    img_tensor = einops.rearrange(img_tensor, "n h w c -> n c h w").contiguous()
    ref_tensor = einops.rearrange(ref_tensor, "n h w c -> n c h w").contiguous()

    txt = [''] * 10
    batch_input = {}
    batch_input['img_gt'] = img_tensor * 2 - 1.0
    batch_input['ref_gt'] = ref_tensor
    batch_input['txt'] = txt
    


    # encoding and decoding
    with torch.cuda.amp.autocast():
        samples = model.log_images(batch = batch_input, sample_steps = steps) # samples['img_gt'] [0,1], samples['samples'] [0,1], samples['bpp']
        metrics_batch_total = cal_metrics_and_save(samples, batch['filenames'], recon_path, batch['refs'])

    return metrics_batch_total


def get_batch_list(input_dir, batchsize):
    assert os.path.isdir(input_dir)

    dir_list = [os.path.join(input_dir, item) for item in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, item))]
    if  dir_list[0].split('/')[-1] == 'ref_256':
        path_ref = dir_list[0]
        path_img = dir_list[1]
    else:
        path_ref = dir_list[1]
        path_img = dir_list[0]
    
    list_ref = [os.path.join(path_ref, item) for item in os.listdir(path_ref)]
    list_img = [os.path.join(path_img, item) for item in os.listdir(path_img)]
    assert len(list_ref) == len(list_img)
    img_num = len(list_ref)
    list_ref.sort()
    list_img.sort()

    batch_list = []
    batch_dict = {}
    batch_dict['imgs'] = []
    batch_dict['refs'] = []
    batch_dict['filenames'] = []
    for idx in range(img_num):
        file_name =  os.path.relpath(list_img[idx], path_img) # ###.png
        img = Image.open(list_img[idx]).convert("RGB")
        ref = Image.open(list_ref[idx]).convert("RGB")
        batch_dict['imgs'].append(img)
        batch_dict['refs'].append(ref)
        batch_dict['filenames'].append(file_name)
        if (idx+1) % batchsize == 0 or idx==(img_num-1):
            batch_list.append(batch_dict.copy())
            batch_dict['imgs'] = []
            batch_dict['refs'] = []
            batch_dict['filenames'] = []
    
    return batch_list





def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    args.device = check_device(args.device)
    ckpt_name = args.ckpt.split('/')[-1]
    output_path = f'{args.output_root}/{ckpt_name}'
    os.makedirs(output_path,exist_ok=True)
    
    # model init
    print('model init...')
    model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
    print('model init over.')

    # model loading
    print('loading state dict...')
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    print('loading over.')

    model.hyper_encoder.update(force=True)

    model.freeze()
    model.to(args.device)
    
    # test set loading
    batch_list = get_batch_list(args.input_path, args.batchsize)

    # total metrics
    results = defaultdict(float)
    for i, batch in enumerate(batch_list): 
        print('processing batch {}:'.format(i+1))
        metrics_batch_total = inference_batch(model, batch,output_path,
            steps=args.steps,
        )

        for k, v in metrics_batch_total.items():
            results[k] += v


    img_paths = os.path.join(args.input_path, 'hr_256')
    img_num = len(os.listdir(img_paths))
    for k, v in results.items():
        results[k] = v / img_num


    # calculating fid and kid
    results['fid'],results['kid'] = get_fid_from_path(args.output_path)

    
    for i in results: 
        results[i] = "{:.4f}".format(results[i])

    output = {
        "results": results,
    }

    with (Path(f"{args.output_path}/average").with_suffix('.json')).open(
        "wb"
    ) as f:
        f.write(json.dumps(output, indent=2).encode())


if __name__ == "__main__":
    main()
