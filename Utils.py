import torch
import torch.nn as nn

# Residual Conv
class Residual_Convs(nn.Module):
    def __init__(self,in_channels,out_channels,residual=False):
        super().__init__()
        
        self.residual=residual
        
        self.block=nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="reflect"),
                                 nn.GroupNorm(1,out_channels),
                                 nn.GELU(),
                                 nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1,padding_mode="reflect"),
                                 nn.GroupNorm(1,out_channels))
        
    
    def forward(self,data):
        
        if self.residual==True:
            return nn.functional.gelu(self.block(data) + data)
        
        else:
            return nn.functional.gelu(self.block(data))
        
        
    
# Downsample          
class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels,embed_dim=256):
        super().__init__()
        
        self.down=nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
                                Residual_Convs(in_channels=in_channels,out_channels=in_channels,residual=True),
                                Residual_Convs(in_channels=in_channels,out_channels=out_channels))
        
        self.embedding_t=nn.Sequential(nn.SiLU(),
                                       nn.Linear(embed_dim,out_channels))
    
    def forward(self,data,t):
        
        x=self.down(data)
        embed_t=self.embedding_t(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1]) # time embeddingi 2 boyutlu olacagı için 4 boyuta çekiyoruz
        
        return x + embed_t # time degerine embedding uygulayıp cıktıya gömüyoruz(topluyoruz) 


# Upsample
class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels,embed_dim=256):
        super().__init__()
        
        self.upsample=nn.Upsample(scale_factor=2)
        
        self.up=nn.Sequential(Residual_Convs(in_channels=in_channels,out_channels=in_channels,residual=True),
                                Residual_Convs(in_channels=in_channels,out_channels=out_channels))
        
        self.embedding_t=nn.Sequential(nn.SiLU(),
                                       nn.Linear(embed_dim,out_channels))
    
    def forward(self,data,skip_data,t):
        x=self.upsample(data)
        x=self.up(torch.cat([x,skip_data],dim=1))
        embed_t=self.embedding_t(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1]) # time embeddingi 2 boyutlu olacagı için 4 boyuta çekiyoruz
        
        return x + embed_t # time degerine embedding uygulayıp cıktıya gömüyoruz(topluyoruz) 
        



# Multihead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,batch_size,d_k=64):
        super().__init__()
        
        self.batch_size=batch_size
        self.d_model=d_model
        self.num_heads=d_model//d_k
        self.d_k=d_k
        
    # Multi Head Attention :
        self.query=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model) 
        self.key=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)     
        self.value=torch.nn.Linear(in_features=self.d_model,out_features=self.d_model)
        self.concat_scaled_dot_product=torch.nn.Linear(d_model,d_model) 
            
    
    def Multi_Head_Attention(self,data):        
            
        query=self.query(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3) 
        key=self.key(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3)    
        value=self.value(data).reshape(self.batch_size,-1,self.num_heads,self.d_k).permute(0,2,1,3) 
            
        dot_product=torch.matmul(query,torch.transpose(key,-2,-1))/((self.d_model/self.num_heads)**1/2) 
        scaled_dot=torch.nn.functional.softmax(dot_product,dim=-1)   
        
        scaled_dot=torch.matmul(scaled_dot,value) 
        
        scaled_dot=scaled_dot.permute(0,2,1,3)
        
        concat_scaled_dot_product=scaled_dot.reshape_as(data)  # concatinate
        
        concat_scaled_dot_product=self.concat_scaled_dot_product(concat_scaled_dot_product) 
        
        return concat_scaled_dot_product
    
    def forward(self,data):
        
        data_new=data.reshape(self.batch_size,data.shape[1],-1).permute(0,2,1) # (batch_size,img_size*img_size,channels)
        mhe_out=self.Multi_Head_Attention(data=data_new)
        mhe_out=mhe_out.permute(0,2,1).reshape_as(data) 
        out=nn.functional.layer_norm((mhe_out+data),mhe_out.shape)
        
        return out