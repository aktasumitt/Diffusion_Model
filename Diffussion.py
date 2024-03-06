import torch

class Diffusion():
    def __init__(self,batch_size,beta_start=0.0001,beta_end=0.02,n_steps=1000,img_size=64,devices="cuda") :
        super(Diffusion,self).__init__()

        self.img_size=img_size
        self.devices=devices
        self.n_steps=n_steps
        self.batch_size=batch_size
        
        self.beta=torch.linspace(beta_start,beta_end,n_steps).to(devices) # eşit aralıklarla 1000 boyutlu beta tensoru olusturma        
        
        self.alpha=1-self.beta # formulden geliyor 
        
        self.alpha_hat=torch.cumprod(self.alpha,0).to(devices) # kümülatif çarpım yani kendinden onceki degerlerle çarpımı ex: (a2=a0*a1*a2), (a3=a0*a1*a2*a3), (a4=a3*a2*a1*a0)
        
    
    def apply_noise_to_img(self,x,t):
        sqrt_alpha=torch.sqrt(self.alpha_hat[t])[:,None,None,None]             # img 4 boyutlu olacagı için 3 boyut daha ekliyoruz
        sqrt_alpha_hat=torch.sqrt(self.alpha_hat[t])[:,None,None,None]   # img 4 boyutlu olacagı için 3 boyut daha ekliyoruz
        
        epsilon=torch.randn_like(x).to(self.devices) # img boyutlarında normal distrubution olusturuyoruz
        
        return sqrt_alpha*x+sqrt_alpha_hat*epsilon,epsilon  # formule gore noise'i img'a her (t) adımda  ekliyoruz
    
    def create_timestep(self):
        return torch.randint(1,self.n_steps,size=(self.batch_size,)).to(self.devices)
        
    # test için sample olusturup egitiyoruz
    def sampling(self,model,labels): 
        
        x=torch.randn((self.batch_size,3,self.img_size,self.img_size)).to(self.devices) # random img noise olusturduk
        with torch.no_grad():
            for i in reversed(range(0,self.n_steps)): # adımlarda sondan başa dogru gidilecegi için reverse alıyoruz
                
                t=torch.ones((self.batch_size),dtype=torch.int)*i # i.adımda batch_size kadar time olusturuyoruz
                self.t=t
                beta=self.beta[t][:,None,None,None]
                alpha=self.alpha[t][:,None,None,None]
                alpha_hat=self.alpha_hat[t][:,None,None,None]
                
                predicted_noise=model(x,t,labels) # model ile resmin içerisindeki noiseyi tahmin etmemiz gerekiyor
                
                # random noise olusturuyoruz
                if i==0:
                    noise=torch.zeros_like(x).to(self.devices)
                else:
                    noise=torch.randn_like(x).to(self.devices)
                
                # Formulu uygulayarak gürültüyü bu formul ile çıkartmaya çalısıyoruz.
                x = 1 / torch.sqrt(alpha)* (x- ((1-alpha)) / (torch.sqrt(1-alpha_hat))*predicted_noise) + torch.sqrt(beta)*noise
        
        new_x=(new_x.clamp(-1,1)+1)/2
        new_x=(new_x*255).type(torch.uint8)
            
        return new_x
