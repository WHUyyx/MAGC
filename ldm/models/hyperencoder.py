import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import ste_round
from compressai.models.base import CompressionModel
from compressai.ans import BufferedRansEncoder, RansDecoder
from ldm.spade.architecture import SPADEResnetBlock 


from model.layers import conv3x3, myResnetBlock, myDownsample, myUpsample, LayerNorm, myBlock


class HyperEncoder(CompressionModel):
    def __init__(self, N=128, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = 4 
        self.hy_out_dim = 64 
        self.num_slices = 8
        self.channel_per_slice = int(self.hy_out_dim / self.num_slices)
        self.max_support_slices = -1 
        # self.use_spade = False
        self.use_spade = True

        self.ga_1 =  nn.Sequential(
            conv3x3(4, N),    
            myResnetBlock(N, N),
            myResnetBlock(N, N),
        )        
        self.ga_2 =  nn.Sequential(
            myDownsample(N),        
            myResnetBlock(N, N),
            myResnetBlock(N, N),
        )
        self.ga_3 =  nn.Sequential(
            myDownsample(N),        
            myResnetBlock(N, N),
            myResnetBlock(N, N),
        )
        self.ga_4 =  conv3x3(N, self.hy_out_dim)
        
        if self.use_spade:
            self.ga_sem1 =  SPADEResnetBlock(N, N)
            self.ga_sem2 =  SPADEResnetBlock(N, N)
            self.ga_sem3 =  SPADEResnetBlock(N, N)
        else:
            self.ga_sem1 =  myResnetBlock(N, N)
            self.ga_sem2 =  myResnetBlock(N, N)
            self.ga_sem3 =  myResnetBlock(N, N)

        self.gs_1 =  nn.Sequential(
            conv3x3(self.hy_out_dim, N),
            myResnetBlock(N, N),
            myResnetBlock(N, N),
        )        
        self.gs_2 =  nn.Sequential(
            myUpsample(N),  
            myResnetBlock(N, N),
            myResnetBlock(N, N),
        )
        self.gs_3 =  nn.Sequential(
            myUpsample(N),  
            myResnetBlock(N, N),
            myResnetBlock(N, N),
        )
        self.gs_4 =  conv3x3(N, 4)
        if self.use_spade:
            self.gs_sem1 =  SPADEResnetBlock(N, N)
            self.gs_sem2 =  SPADEResnetBlock(N, N)
            self.gs_sem3 =  SPADEResnetBlock(N, N)
        else:
            self.gs_sem1 =  myResnetBlock(N, N)
            self.gs_sem2 =  myResnetBlock(N, N)
            self.gs_sem3 =  myResnetBlock(N, N)



        self.ha_1 = nn.Sequential(
            conv3x3(self.hy_out_dim, N),
            myResnetBlock(N, N),
            myDownsample(N),
            myResnetBlock(N, N),
            conv3x3(N,self.hy_out_dim)
        )

   
        self.hs_1 = nn.Sequential(
            conv3x3(self.hy_out_dim, N),
            myResnetBlock(N, N),
            myUpsample(N),
            myResnetBlock(N, N),
            conv3x3(N, 2 * self.hy_out_dim)
        )


        self.entropy_bottleneck = EntropyBottleneck(self.hy_out_dim)
        self.gaussian_conditional = GaussianConditional(None)

        self.sem_encoder =  nn.Sequential( 
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
        )   

        self.sem_conv1 = nn.Sequential(
            conv3x3(192,192,2),
            LayerNorm(192),
            nn.LeakyReLU()
        )   

        self.sem_conv2 = nn.Sequential(
            conv3x3(192,192,2),
            LayerNorm(192),
            nn.LeakyReLU()
        )   


        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(self.hy_out_dim + self.channel_per_slice * i, 128, 1), 
                nn.GELU(),  
                conv3x3(128, 64, 1),
                nn.GELU(),
                conv3x3(64, 32, 1),
                nn.GELU(),
                conv3x3(32, 16, 1),
                nn.GELU(),
                conv3x3(16, self.channel_per_slice, 1),
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv3x3(self.hy_out_dim + self.channel_per_slice * i, 128, 1), 
                nn.GELU(),  
                conv3x3(128, 64, 1),
                nn.GELU(),
                conv3x3(64, 32, 1),
                nn.GELU(),
                conv3x3(32, 16, 1),
                nn.GELU(),
                conv3x3(16, self.channel_per_slice, 1),
            ) for i in range(8)
        )



    def ga(self,y,sem):
        y = self.ga_1(y)
        y = self.ga_sem1(y, sem[0])
        y = self.ga_2(y)
        y = self.ga_sem2(y, sem[1])
        y = self.ga_3(y)
        y = self.ga_sem3(y, sem[2])
        y = self.ga_4(y)
        return y
    
    def ga_without_sem(self,y):
        y = self.ga_1(y)
        y = self.ga_sem1(y)
        y = self.ga_2(y)
        y = self.ga_sem2(y)
        y = self.ga_3(y)
        y = self.ga_sem3(y)
        y = self.ga_4(y)
        return y
    
    def gs(self,y ,sem):
        y = self.gs_1(y)
        y = self.gs_sem1(y, sem[2])
        y = self.gs_2(y)
        y = self.gs_sem2(y, sem[1])
        y = self.gs_3(y)
        y = self.gs_sem3(y, sem[0])
        y = self.gs_4(y)
        return y
    
    def gs_without_sem(self,y):
        y = self.gs_1(y)
        y = self.gs_sem1(y)
        y = self.gs_2(y)
        y = self.gs_sem2(y)
        y = self.gs_3(y)
        y = self.gs_sem3(y)
        y = self.gs_4(y)
        return y
    
    def ha(self, y):
        y = self.ha_1(y)
        return y

    def hs(self, y):
        y = self.hs_1(y)
        return y

    def get_sem_latent(self,sem):
        sem_latents = []
        sem = self.sem_encoder(sem)
        sem_latents.append(sem)
        sem = self.sem_conv1(sem)
        sem_latents.append(sem)
        sem = self.sem_conv2(sem)
        sem_latents.append(sem)
        return sem_latents



    def forward(self, x, sem):
        # encoding
        if sem is not None:
            sem_latents = self.get_sem_latent(sem) # b 192 32 32
            y = self.ga(x, sem_latents)
        else:
            y = self.ga_without_sem(x)

        y_shape = y.shape[2:]
        z = self.ha(y)
        _, z_likelihoods = self.entropy_bottleneck(z)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        gaussian_params = self.hs(z_hat)
        latent_scales, latent_means = gaussian_params.chunk(2,1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            y_hat_slices.append(y_hat_slice)


        # decoding
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        if sem is not None:
            x_hat = self.gs(y_hat, sem_latents) 
        else:
            x_hat = self.gs_without_sem(y_hat) 
            
        return {
            'y_hat': x_hat,
            # 'y_hat': x,
            "likelihoods": [y_likelihoods, z_likelihoods], # calculating bpp_loss
        }
    





        
    
    def hyper_compress(self,x,sem): 
        batch_size = x.size(0)


        # encoding
        if sem is not None:
            sem_latents = self.get_sem_latent(sem) 
            y = self.ga(x, sem_latents)
        else:
            y = self.ga_without_sem(x)


        y_shape = y.shape[2:]
        z = self.ha(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.hs(z_hat)
        latent_scales, latent_means = gaussian_params.chunk(2,1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        y_strings = []
        symbols_list_single = []
        indexes_list_single = []


        for i in range(batch_size):
            symbols_list_single.append([])
            indexes_list_single.append([])

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu


            for i in range(batch_size):
                symbols_list_single[i].extend(y_q_slice[i,:,:,:].reshape(-1).tolist())
                indexes_list_single[i].extend(index[i,:,:,:].reshape(-1).tolist())


            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)


        for i in range(batch_size):
            encoder.encode_with_indexes(symbols_list_single[i], indexes_list_single[i], cdf, cdf_lengths, offsets)
            y_string = encoder.flush()
            y_strings.append(y_string)

        # decoding
        y_hat = torch.cat(y_hat_slices, dim=1)
        if sem is not None:
            x_hat = self.gs(y_hat, sem_latents) 
        else:
            x_hat = self.gs_without_sem(y_hat) 


        return {
            'y_hat': x_hat,
            # 'y_hat': x,
            'z_strings':[y_strings, z_strings], 
        }
        

