import torch
import tqdm


class Diffussion():
    def __init__(self,beta_start,beta_end,n_timesteps,img_size,devices:None):
        
        self.n_timesteps=n_timesteps
        self.img_size=img_size
        self.devices=devices
        
        # For the formula to aplly noising and denoising process
        self.beta=torch.linspace(beta_start,beta_end,n_timesteps).to(self.devices)
        self.alpha=1-self.beta
        self.alpha_hat=torch.cumprod(self.alpha,dim=0)
        
    def Noising_to_Image(self,x,t):
        
        sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None,None,None] # Boyutlandırma
        sqrt_one_minus_alpha_hat=torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
        
        noise=torch.randn_like(x)
        
        noisy_img=(sqrt_alpha_hat*x)+(sqrt_one_minus_alpha_hat*noise)
        
        return noisy_img,noise
    
    def Random_Timesteps(self,batch_size): ### ANLAMADIM
        return torch.randint(1,self.n_timesteps,(batch_size,)).to(self.devices)
    
    
    def Denoising(self,model:None,batch_size:None,labels:None): # Test the model with random noisy img
        prog_bar=tqdm.tqdm(range(self.n_timesteps),"Prediction Image Step")
        model.eval()
        x=torch.randn(batch_size,3,self.img_size,self.img_size).to(self.devices)
        
        for i in enumerate(reversed(range(1,self.n_timesteps))):
            
            T=(torch.ones(batch_size)*i).long().to(self.devices)
    
            predicted_noise=model(x,T,labels)
            
            # CFG predicted_noise. This process about, if we train conditional, after we need to predict uncoditional.
            # We use torch lerp to aproach conditional prediction from unconditional smoothly with 3 scale factor
            if labels!=None:
                predicted_noise_unc=model(x,T,None)
                predicted_noise=torch.lerp(predicted_noise_unc,predicted_noise,3) 
            
            beta=self.beta[T][:,None,None,None]
            alpha=self.alpha[T][:,None,None,None]
            alpha_hat=self.alpha_hat[T][:,None,None,None]
            
            noise=(torch.randn_like(x) if i>1 else torch.zeros_like(x)).to(self.devices)
            
            x = (1/alpha_hat) * (x-((1-alpha)/torch.sqrt(1-alpha_hat))*predicted_noise) +(torch.sqrt(beta)*noise)
            prog_bar.update(1)
        
        prog_bar.close()
        
        model.train()
        
        x=(x.clamp(-1,1) + 1) / 2
        x=(x*255).type(torch.uint8)
        return x
