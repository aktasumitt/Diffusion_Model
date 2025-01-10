import torch.nn as nn
import torch

# Double convolutional layer for the downsampling and upsampling blocks
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None,residual=False):
        super().__init__()
        self.residual=residual
        if mid_channels == None:
            mid_channels=out_channels
        
        self.conv=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=mid_channels,kernel_size=3,padding=1),
                                nn.GroupNorm(1,mid_channels),
                                nn.GELU(),
                
                                nn.Conv2d(in_channels=mid_channels,out_channels=out_channels,kernel_size=3,padding=1),
                                nn.GroupNorm(1,out_channels))     
    
    
    def forward(self,x):
        
        out=self.conv(x)
        
        if self.residual==True:
            return nn.functional.gelu(x+out)
        
        else:
            return out


        
# Down sampling block for the encoder of Unet model    
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,embed_dim=256):
        super().__init__()
        self.down=nn.Sequential(nn.MaxPool2d(2),
                                DoubleConv(in_channels=in_channels,out_channels=in_channels,residual=True),
                                DoubleConv(in_channels=in_channels,out_channels=out_channels))
        
        # embedding for time
        self.embed=nn.Sequential(nn.SiLU(),
                                 nn.Linear(embed_dim,out_channels))
    
    def forward(self,x,t):
        
        out_d=self.down(x)        
        
        t_embed=self.embed(t)[:,:,None,None].repeat(1,1,out_d.shape[-2],out_d.shape[-1])
        
        return out_d+t_embed


# Down sampling block for the decoder of Unet model
class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,embed_dim=256):
        super().__init__()
        
        self.up=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        
        self.up_block=nn.Sequential(DoubleConv(in_channels=in_channels,out_channels=in_channels,residual=True),
                                    DoubleConv(in_channels=in_channels,out_channels=out_channels,mid_channels=out_channels//2))
        
        self.embedding=nn.Sequential(nn.SiLU(),
                                    nn.Linear(embed_dim,out_channels))
        
        
    def forward(self,x,x_enc,t):
        
        x=self.up(x)
        out_up=self.up_block(torch.cat([x,x_enc],dim=1))
        t_embed=self.embedding(t)[:,:,None,None].repeat(1,1,out_up.shape[-2],out_up.shape[-1])
        
        return out_up+t_embed
        
    

# Multihead attention for between the blocks
class MultiheadAttention(nn.Module):
    
    def __init__(self,channel_size,size):
        super().__init__()
        
        self.size=size
        self.channel_size=channel_size
        
        self.layer_norm=nn.LayerNorm([channel_size])
        
        self.mhe=nn.MultiheadAttention(channel_size,num_heads=4,batch_first=True)
        
        self.feed_forward=nn.Sequential(nn.LayerNorm([channel_size]),
                                        nn.Linear(channel_size,channel_size),
                                        nn.GELU(),
                                        nn.Linear(channel_size,channel_size))
        
        
    def forward(self,x):
        x=x.view(-1,self.channel_size,self.size*self.size).permute(0,2,1)
        
        x_ln=self.layer_norm(x)
        
        out_mhe,_=self.mhe(x_ln,x_ln,x_ln)
        
        out_mhe=out_mhe+x # residual
        
        out=self.feed_forward(out_mhe) + out_mhe # residual
        
        return out.permute(0,2,1).view(-1,self.channel_size,self.size,self.size)