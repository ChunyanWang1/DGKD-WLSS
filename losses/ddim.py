# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import math
from torch import nn
import torch.nn.functional as F
from .scheduling_ddim import DDIMScheduler
import time
import torch.distributed as dist

class DDIM(nn.Module):

    def __init__(
            self,
            feat_channels,
            channels,
            kernel_size=3,
            inference_steps=5,#5,
            num_train_timesteps=1000,
    ):
        super().__init__()
        #self.trans = nn.Conv2d(feat_channels, channels, 1)
        self.model = ScheduledCNNRefine(channels_in=channels, channels_noise=channels, kernel_size=kernel_size)
        self.diffusion_inference_steps = inference_steps
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)
        self.proj = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.BatchNorm2d(channels))
        
    def forward(self, feat, gt_feat,return_loss=True):#feat_aux, 
        """
        fp: List[Tensor]
        depth_map: Tensor with shape bs, 1, h, w
        depth_mask: Tensor with shape bs, 1, h, w
        """
        #feat = self.trans(feat)
        refined_feat = self.pipeline(
            batch_size=feat.shape[0],
            device=feat.device,
            dtype=feat.dtype,
            shape=feat.shape[1:],
            feat=feat,
            gt_feat=gt_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
       
        refined_feat = self.proj(refined_feat)
        if not return_loss:
            return refined_feat
        
        ddim_loss = self.ddim_loss(feat, gt_feat)
        return refined_feat, ddim_loss, gt_feat

    def ddim_loss(self, feat, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        #print("timesteps:",timesteps)
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)#gt_feat
        noise_pred = self.model(noisy_images, timesteps, gt_feat)
        

        loss = F.mse_loss(noise_pred, noise)

        return loss


class SpatialAutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels, downsample=4):
        super().__init__()
        self.encoder = nn.Sequential(
            *[Bottleneck(channels if i == 0 else latent_channels, latent_channels, stride=2) for i in range(int(math.log(downsample, 2)))]
        )
        self.decoder = nn.Sequential(
            *[Bottleneck(latent_channels, channels if i == int(math.log(downsample, 2)) -1 else latent_channels, stride=1) for i in range(int(math.log(downsample, 2)))]
        )

    def forward(self, x):
        _, _, h, w = x.shape
        hidden = self.encoder(x)
        hidden_ = hidden
        for i in range(len(self.decoder)):
            if i != len(self.decoder) - 1:
                hidden_ = F.interpolate(hidden_, scale_factor=2, mode='nearest')
            else:
                hidden_ = F.interpolate(hidden_, size=(h, w), mode='nearest')
            hidden_ = self.decoder[i](hidden_)
        return hidden, hidden_

    def forward_encoder(self, x):
        return self.encoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            #nn.BatchNorm2d(channels // 4),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(channels//4, latent_channels, 1, bias=False),
            nn.BatchNorm2d(latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, padding=0),
            #nn.BatchNorm2d(channels // 4),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(channels//4, channels, 1, bias=True),
            #nn.BatchNorm2d(latent_channels)
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)


#from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
class CNNDDIMPipiline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler, noise_adapter=None, solver='ddim'):#gaussian1,gaussian2,
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter
        self._iter = 0
        self.solver = solver
        #self.noise = nn.Parameter(torch.zeros(1, 512, 7, 7).cuda()) if model.kernel_size == 3 else nn.Parameter(torch.zeros(1, 1000, 1, 1).cuda())
        #self.noise = torch.randn(1, 256, 7, 7).cuda() if model.kernel_size == 3 else torch.randn(1, 1000, 1, 1).cuda()
        if solver == 'dpm':
            noise_schedule = NoiseScheduleVP(schedule='discrete', alphas_cumprod=scheduler.alphas_cumprod)
            model_fn = model_wrapper(model, noise_schedule)
            self.dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type='dpmsolver++')
    


    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            feat,
            gt_feat,
            #feat_aux,
            generator = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            proj = None
    ):
        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{self.device}")` instead.'
            )
            raise RuntimeError(
                "generator.device == 'cpu'",
                "0.11.0",
                message,
            )
            generator = None

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        if self.noise_adapter is not None:
            noise = torch.randn(image_shape, device=device, dtype=dtype)#+feat_aux
            timesteps = self.noise_adapter(feat)
            if self._iter % 200 == 0: # and dist.get_rank() == 0:
                print(timesteps.detach()[:10])
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
        else:
            image = feat

        if self.solver == 'dpm':
            image = self.dpm_solver.sample(image, steps=5, order=3, skip_type='time_uniform', method='singlestep', t_start=0.5)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps*2)

        for t in self.scheduler.timesteps[len(self.scheduler.timesteps)//2:]:
            # timesteps 选择了20步
            # 1. predict noise model_output
            noise_pred = self.model(image, t.to(device), gt_feat)

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                noise_pred, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample']
            #vis(proj(image).detach(), f'vis/sr_{feat.shape[2]}_{t.item()}.png')    
                
        self._iter += 1        
        return image


class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            #nn.GroupNorm(4, in_channels // reduction),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            #nn.GroupNorm(4, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            #nn.GroupNorm(4, out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, stride=1):
        super().__init__()
        mid_channels = int(in_channels / reduction)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            #nn.GroupNorm(4, mid_channels),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, stride=stride),
            nn.BatchNorm2d(in_channels // reduction),
            #nn.GroupNorm(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1),
            #nn.GroupNorm(4, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=stride) if (stride != 1 or in_channels != out_channels) else nn.Identity()

    def forward(self, x):
        out = self.block(x)
        return out + self.residual(x)


'''
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            #nn.GroupNorm(4, in_channels // reduction),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            #nn.GroupNorm(4, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            #nn.GroupNorm(4, out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x
'''


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ScheduledCNNRefine(nn.Module):
    def __init__(self, channels_in, channels_noise, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        '''
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, channels_in * 4, 1, stride=1),
            nn.GroupNorm(4, channels_in * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in * 4, channels_in, 1, stride=1),
            nn.GroupNorm(4, channels_in),
            nn.ReLU(inplace=True),
        )
        '''
        # TODO: add class embedding

        self.time_embedding = nn.Embedding(1280, channels_in)#channels_in*2
        #self.class_embedding = nn.Embedding(num_classes, channels_in)

        '''
        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // 4, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(4, channels_in // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in // 4, channels_noise, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(4, channels_noise),
            nn.ReLU(inplace=True),
        )
        '''
        if kernel_size == 3:
            self.pred = nn.Sequential(
                Bottleneck(channels_in, channels_in),#(channels_in*2, channels_in)
                Bottleneck(channels_in, channels_in),
                #Bottleneck(channels_in, channels_in),
                nn.Conv2d(channels_in, channels_in, 1),
                nn.BatchNorm2d(channels_in)
            )
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),#(channels_in*2, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1)
            )

    def forward(self, noisy_image, t, feat=None):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image
        if t.numel() == 1:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None] #+ self.class_embedding(label)[..., None, None]
            # feat = feat + self.time_embedding(t)[None, :, None, None]
            # t 如果本身是一个值，需要扩充第一个bs维度 (这个暂时不适用)
        else:
            feat = feat + self.time_embedding(t)[..., None, None] #+ self.class_embedding(label)[..., None, None]
        feat = feat #+ self.noise_embedding(noisy_image)

        ret = self.pred(feat)

        return ret


if __name__ == '__main__':
    import copy
    # test
    feat = torch.randn(2, 256, 1, 1).cuda()
    feat_t = torch.randn(2, 256, 1, 1).cuda()

    channels_in = 256
    model = nn.Sequential(
        nn.Conv2d(channels_in, channels_in * 4, 1, stride=1),
        nn.BatchNorm2d(channels_in * 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels_in * 4, channels_in, 1, stride=1),
        nn.BatchNorm2d(channels_in),
    )
    model.cuda()
    model.train()
    ori_state = copy.deepcopy(model.state_dict())

    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for it in range(1000):
        optim.zero_grad()
        p_feat = model(feat)
        kd_loss = F.mse_loss(p_feat, feat_t)
        print(f'kd loss: {kd_loss.item():.4f}')
        loss = kd_loss
        loss.backward()
        optim.step()

    model.load_state_dict(ori_state)
    ddim = DDIM(256, 256).cuda()
    ddim.train()
    print(ddim)
    optim = torch.optim.SGD(set(list(ddim.parameters()) + list(model.parameters())), lr=0.1, momentum=0.9)
    for it in range(1000):
        optim.zero_grad()
        p_feat = model(feat)
        r_feat, ddim_loss = ddim(p_feat, feat_t)
        kd_loss = F.mse_loss(r_feat, feat_t)
        print(f'kd loss: {kd_loss.item():.4f} ddim loss: {ddim_loss.item():.4f}')
        loss = ddim_loss + kd_loss
        loss.backward()
        optim.step()
