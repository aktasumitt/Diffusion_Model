import torch
import matplotlib.pyplot as plt

def Prediction(Prediction:bool,label:str,class_to_idx,Diffussion_Model,Model,batch_size,devices):
    
    label_idx=class_to_idx[label]
    labels=torch.tensor([label_idx]).repeat(batch_size,).to(devices)
    
    
    img_pred=Diffussion_Model.Denoising(Model,batch_size,labels)
    
     
    for i in range(1,7):
        
        sub=plt.subplot(1,6,i)
        sub.imshow(img_pred[i].permute(1,2,0))
        plt.xticks([]) 
        plt.yticks([])
    
    plt.xlabel(label)
    plt.show()  
        