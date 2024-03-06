from Utils import Residual_Convs,Downsample,Upsample,MultiHeadAttention
import torch
import torch.nn as nn

class U_Net(nn.Module):
    def __init__(self,channels_size,batch_size,embedding_dim,label_size,initial_filter_size=64,devices="cuda"):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.devices=devices
        
        # ENCODER:
        
        
        self.initial_layer=Residual_Convs(in_channels=channels_size,out_channels=initial_filter_size)
        
        self.d1=Downsample(in_channels=initial_filter_size,out_channels=initial_filter_size*2)
        self.att1=MultiHeadAttention(d_model=initial_filter_size*2,batch_size=batch_size)
        
        self.d2=Downsample(in_channels=initial_filter_size*2,out_channels=initial_filter_size*4)
        self.att2=MultiHeadAttention(d_model=initial_filter_size*4,batch_size=batch_size)
        
        self.d3=Downsample(in_channels=initial_filter_size*4,out_channels=initial_filter_size*8)
        self.att3=MultiHeadAttention(d_model=initial_filter_size*8,batch_size=batch_size)
        
        self.d4=Downsample(in_channels=initial_filter_size*8,out_channels=initial_filter_size*8)
        self.att4=MultiHeadAttention(d_model=initial_filter_size*8,batch_size=batch_size)
        
        # LATENT SPACE:
        self.last_down=nn.MaxPool2d(2,2)
        self.latent1=Residual_Convs(in_channels=initial_filter_size*8,out_channels=initial_filter_size*8)
        self.latent2=Residual_Convs(in_channels=initial_filter_size*8,out_channels=initial_filter_size*8)
        self.latent3=Residual_Convs(in_channels=initial_filter_size*8,out_channels=initial_filter_size*8)
        
        # DECODER:
                
        self.up1=Upsample(in_channels=initial_filter_size*16,out_channels=initial_filter_size*8)
        self.d_att1=MultiHeadAttention(d_model=initial_filter_size*8,batch_size=batch_size)
        
        self.up2=Upsample(in_channels=initial_filter_size*16,out_channels=initial_filter_size*4)
        self.d_att2=MultiHeadAttention(d_model=initial_filter_size*4,batch_size=batch_size)
        
        self.up3=Upsample(in_channels=initial_filter_size*8,out_channels=initial_filter_size*2)
        self.d_att3=MultiHeadAttention(d_model=initial_filter_size*2,batch_size=batch_size)
        
        self.up4=Upsample(in_channels=initial_filter_size*4,out_channels=initial_filter_size)
        self.d_att4=MultiHeadAttention(d_model=initial_filter_size,batch_size=batch_size) 
        
        self.upsample_last=nn.Upsample(scale_factor=2)
        self.out_layer=Residual_Convs(in_channels=initial_filter_size,out_channels=channels_size)
        
        # Embedding
        self.embedding_label=nn.Embedding(num_embeddings=label_size,embedding_dim=embedding_dim)
    
    # Positional embedding
    def Positional_encoding(self, t):
        
        inverse = 1 / 10000 ** (torch.arange(1, self.embedding_dim, 2, dtype=torch.float) / self.embedding_dim).to(self.devices)        
        
        # Tekrarlama işlemi
        repeated_t = t.unsqueeze(-1).repeat(1, (self.embedding_dim // 2)).to(self.devices)  
        
        
        # pos_A ve pos_B'nin oluşturulması
        pos_A = torch.sin(repeated_t * inverse)
        pos_B = torch.cos(repeated_t * inverse)
        
        return torch.cat([pos_A, pos_B], dim=-1)


    def forward(self,data,t,label):
        
        # Embedding for the conditioning
        embed_label=self.embedding_label(label)
        t=self.Positional_encoding(t)
        t+=embed_label

        # encoder
        x=self.initial_layer(data)
        
        d1=self.d1(x,t)
        d1=self.att1(d1)
        
        d2=self.d2(d1,t)
        d2=self.att2(d2)
        
        d3=self.d3(d2,t)
        d3=self.att3(d3)
        
        d4=self.d4(d3,t)
        d4=self.att4(d4)
        
        
        # Latent Space
        down_last=self.last_down(d4)
        l1=self.latent1(down_last)
        l2=self.latent1(l1) 
        l3=self.latent1(l2)
             
                
        # Decoder
        u1=self.up1(l3,d4,t)
        u1=self.d_att1(u1)
        
        u2=self.up2(u1,d3,t)
        u2=self.d_att2(u2)
        
        u3=self.up3(u2,d2,t)
        u3=self.d_att3(u3)
        
        u4=self.up4(u3,d1,t)
        u4=self.d_att4(u4)
        
        up=self.upsample_last(u4)
        out=self.out_layer(up)
        
        return out
