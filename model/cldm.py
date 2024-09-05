import torch
import json
from pathlib import Path

from ldm.modules.diffusionmodules.util import timestep_embedding

from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from utils.common import frozen_module
from .spaced_sampler import SpacedSampler

import numpy as np
import os
from torchvision import transforms
from PIL import Image


from model.adapters import Adapter_XL
from ldm.models.hyperencoder import HyperEncoder
from collections import defaultdict
from cal_metrics.iqa import single_iqa


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, y_hat = None,**kwargs):

        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
        if y_hat is not None:
            x = torch.cat((x, y_hat), dim=1)

        h = x.type(self.dtype) 
        if control is not None:
            for idx, module in enumerate(self.input_blocks):
                if idx == 0:
                    h = module(h, emb, context) 
                elif len(control)!=0 and h.shape == control[0].shape:
                    h = module(h+control.pop(0), emb, context) 
                else:
                    h = module(h, emb, context) 
                hs.append(h) 
        else:
            for idx, module in enumerate(self.input_blocks):
                h = module(h, emb, context) 
                hs.append(h)

        h = self.middle_block(h, emb, context) # 16 1280 4 4

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)



class ControlLDM(LatentDiffusion):

    def __init__(
        self,
        control_key: str,
        sd_locked: bool,
        only_mid_control: bool,
        training_stage: str,
        learning_rate: float,
        *args,
        **kwargs
    ) -> "ControlLDM":
        super().__init__(*args, **kwargs)
        # self.use_control = False
        self.use_control = True

        if self.use_control:
            self.control_model = Adapter_XL()
        else:
            self.control_model = None

        self.control_key = control_key
        self.sd_locked = sd_locked
        self.training_stage = training_stage
        self.only_mid_control = only_mid_control
        self.learning_rate = learning_rate

        self.use_scheduler = True # warmup
        self.set_iqa = False
        self.save_val_images = False
        
        if self.training_stage=='stage1':
            self.training_stage1 = True 
            self.training_stage2 = False 
        if self.training_stage=='stage2':
            self.training_stage1 = False 
            self.training_stage2 = True 


        # hyper_encoder
        self.hyper_encoder = HyperEncoder()
        self.use_spade = self.hyper_encoder.use_spade


        if self.training_stage2: # fixing the encoder side and entropy model
            for name, module in self.hyper_encoder.named_modules():
                if('gs_' not in name) and ('hs_' not in name) and ('.' not in name) and name!='':
                    frozen_module(module)

        self.model.train() 
        first_kernal = 'diffusion_model.input_blocks.0.0.weight'
        for name, param in self.model.named_parameters():
            if 'attn1' in name or 'attn2' in name or 'proj_in' in name or 'proj_out' in name or name == first_kernal:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # tensorboard 
        self.loss_simple = defaultdict(list)

        # image save rootpath
        if self.save_val_images:
            version_list = os.listdir('save_dir/lightning_logs')
            version_list_int = [int(i[8:]) for i in version_list]
            version_num = sorted(version_list_int)[-1] +1 
            self.val_img_output_rootpath = f'save_dir/img_output/version_{version_num}/'

        # warmup step
        self.warmup_steps = 10000


    def apply_condition_encoder(self, control):
        c_latent_meanvar = self.cond_encoder(control * 2 - 1)
        c_latent = DiagonalGaussianDistribution(c_latent_meanvar).mode() # only use mode
        c_latent = c_latent * self.scale_factor
        return c_latent
    

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        with torch.no_grad():
            x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs) # latent, text embedding
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            # control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float() # dlg, 4 3 256 256
        return x, dict(c_crossattn=[c], c_ref=[control])


    def apply_model(self, x_noisy, t, cond, y_hat, *args, **kwargs): 
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_text_embadding = torch.cat(cond['c_crossattn'], 1) # b 77 1024
        
        if self.use_control == False:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_text_embadding, control=None, only_mid_control=self.only_mid_control, y_hat=y_hat)
        else:
            if cond['c_ref'][0] is None:
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_text_embadding, control=None, only_mid_control=self.only_mid_control, y_hat=y_hat)
            else:
                cont_ref = torch.cat(cond['c_ref'], 1) # b 3 256 256
                control = self.control_model(cont_ref,x_noisy ) # adapter
                control = [c * scale for c, scale in zip(control, [self.scale_factor] * len(control))] 
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_text_embadding, control=control, only_mid_control=self.only_mid_control, y_hat=y_hat) 
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, sample_steps=50):
        # batch['img_gt'] [-1,1]  batch['ref_gt'] 
        log = dict()
        y, c = self.get_input(batch, self.first_stage_key) # y: b 4 32 32, c["c_ref"][0]: 
        c_ref, c_text_embedding = c["c_ref"][0], c["c_crossattn"][0] # ref b 3 256 256 ; text_embedding b 77 1024
        log["img_gt"] = (batch['img_gt'] + 1) / 2 # img_reconstruct b 3 256 256

        if self.use_spade:
            hyper = self.hyper_encoder.hyper_compress(y,batch['ref_gt'])  # hyper['z_string'],  hyper['z_hat']
        else:
            hyper = self.hyper_encoder.hyper_compress(y,None)

        y_hat, z_strings  =  hyper['y_hat'], hyper['z_strings']
        # y_hat, z_strings  =  y, hyper['z_strings']
        log['bpp_list'],log['bpp_list_z'],log['bpp_list_zz'] = self.cal_bpp(z_strings, batch['img_gt'])

        if self.use_spade:
            temp = self.hyper_encoder.forward(y,batch['ref_gt'] )
        else:
            temp = self.hyper_encoder.forward(y,None)

        likelihoods = temp['likelihoods']
        log["bpp_loss"] = self.cal_bpp_loss(likelihoods, batch['img_gt'])

        if self.training_stage1:
            log["y_mse"] = torch.nn.functional.mse_loss(y, y_hat)
            samples = (self.decode_first_stage(y_hat) + 1 )/2 # 0,1
            # samples = (self.decode_first_stage(y) + 1 )/2 # 0,1
            samples = torch.clamp(samples, 0, 1)

        elif self.training_stage2: 
            samples, img_latent = self.sample_log( # b 3 256 256 [0,1] 
                # TODO: remove c_concat from cond
                cond={"c_ref": [c_ref], "c_crossattn": [c_text_embedding]},
                steps=sample_steps,
                y_hat = y_hat, # b 4 32 32 
            )
            log["y_mse"] = torch.nn.functional.mse_loss(y, img_latent) # 二阶段的mse更大
        log["samples"] = samples
        return log



    @torch.no_grad()
    def sample_log(self, cond, steps, y_hat):
        sampler = SpacedSampler(self)

        b, c, h_latent, w_latent = y_hat.shape
        shape = (b, self.channels, 32, 32)

        img_pixel, img_latent = sampler.sample(
            steps, 
            shape, 
            cond['c_ref'][0] if self.use_control else None,
            y_hat = y_hat
        )

        return img_pixel, img_latent


    def lr_lambda_warmup(self, current_step):
        current_step = float(current_step)
        warmup_steps = float(self.warmup_steps)

        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            return 1.0


    def configure_optimizers(self):
        lr = self.learning_rate
        if self.control_model is not None:
            params_list = [
                list(self.control_model.parameters()),
                list(self.hyper_encoder.parameters()),
                list(self.model.parameters())
            ]

        else:
            params_list = [
                list(self.hyper_encoder.parameters()),
                list(self.model.parameters())
            ]            
        params = []
        for i in params_list:
            params += i
        params += list(self.model.diffusion_model.out.parameters()) # out层也加上

        opt = torch.optim.AdamW(params, lr=lr, weight_decay = 0.01) 

        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, self.lr_lambda_warmup)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [opt], [scheduler]
        return opt


    def val_save_batch_img(self,batch, batch_ref):
        batch_gt = batch['img_gt']
        batch_samples = batch['samples']
        bpp_batch = batch['bpp_list']
        bpp_z_batch = batch['bpp_list_z']
        bpp_zz_batch = batch['bpp_list_zz']



        val_batchsize,_,_,_ = batch_gt.size()
        for i in range(val_batchsize):
            gt = batch_gt[i,:,:,:]
            sample = batch_samples[i,:,:,:]
            ref = batch_ref[i,:,:,:]
            
            if self.save_val_images:
                img_num = int(len(os.listdir(self.val_img_output_path))/2)
                save_path = os.path.join(self.val_img_output_path, f'img_{img_num+1}.png')
                save_path_metrics = os.path.join(self.val_img_output_path, f'img_{img_num+1}.json')
            
                img_gt_single = transforms.ToPILImage()(gt.cpu())
                img_rec_single = transforms.ToPILImage()(sample.cpu())
                ref_single = transforms.ToPILImage()(ref.cpu())

                result_img = Image.new('RGB', (img_gt_single.width + img_rec_single.width + ref_single.width  + 20 , img_gt_single.height))
                result_img.paste(img_gt_single, (0, 0))
                result_img.paste(img_rec_single, (img_gt_single.width + 10, 0))
                result_img.paste(ref_single, (img_gt_single.width + img_rec_single.width + 20, 0))
                result_img.save(save_path)


            # calculating metrics
            metrics = self.single_iqa.cal_metrics(gt.unsqueeze(0), sample.unsqueeze(0))
            metrics_single = {}
            metrics_single['bpp_z'] = bpp_z_batch[i]
            metrics_single['bpp_zz'] = bpp_zz_batch[i]
            metrics_single['bpp'] = bpp_batch[i]

            for k in metrics:
                metrics_single[k] = metrics[k]
            if self.save_val_images:
                with Path(save_path_metrics).open("wb") as f:
                    output = {
                        "results": metrics_single,
                    }
                    f.write(json.dumps(output, indent=2).encode())

            for k in metrics_single:
                self.loss_simple[k].append(metrics_single[k])





    def on_validation_epoch_start(self, *args, **kwargs):
        if self.set_iqa == False:
            self.single_iqa = single_iqa(device = self.device)
            self.set_iqa = True

        for k in self.loss_simple:
            self.loss_simple[k] = []

        if self.save_val_images:
            self.val_img_output_path = f'{self.val_img_output_rootpath}/globalstep_{self.global_step}'
            if not os.path.isdir(self.val_img_output_path):
                os.makedirs(self.val_img_output_path, exist_ok = True)

        self.model.eval()
        self.hyper_encoder.update(force=True)
        

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # TODO: 
        # pass
        batch_result = self.log_images( 
            batch = batch,
            sample_steps = 50
        )

        batch_result['samples'] = torch.clamp(batch_result['samples'], min=0,max=1)
        self.val_save_batch_img(batch_result, batch['ref_gt'])
        bpp_loss = batch_result['bpp_loss'].item()
        self.loss_simple['bpp_loss'].append(bpp_loss)

        loss, loss_dict = self.shared_step(batch)
        self.loss_simple['val_loss'].append(loss.item())
        self.loss_simple['y_mse'].append(batch_result['y_mse'].item())




    def on_validation_epoch_end(self, *args, **kwargs):
        metrics_current = dict()
        for i in self.loss_simple:
            if i not in ['val_loss', 'bpp_loss', 'y_mse']:
                metrics_current[i] = np.mean(self.loss_simple[i])
                self.log(i, metrics_current[i])



        val_loss_current = np.mean(self.loss_simple['val_loss']) 
        bpploss_current = np.mean(self.loss_simple['bpp_loss']) 
        ymse_current = np.mean(self.loss_simple['y_mse']) 
        
        self.log('val_loss', val_loss_current)
        self.log('val_bpploss', bpploss_current)
        self.log('val_y_mse', ymse_current)

        for i in metrics_current: 
            metrics_current[i] = "{:.4f}".format(metrics_current[i])
        output = {
            "results": metrics_current,
        }

        if self.save_val_images:
            with (Path(f"{self.val_img_output_path}/eva_result").with_suffix('.json')).open(
                "wb"
            ) as f:
                f.write(json.dumps(output, indent=2).encode())

        self.model.train()
        




