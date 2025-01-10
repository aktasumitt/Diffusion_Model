import torch
import tqdm

# Training Func
def Training(EPOCH,Train_Dataloader,Model,Diffussion_Model,optimizer,loss_fn,Save_Checkpoint_fn,Checkpoints_dir,devices,STARTING_EPOCH=1,Tensorboard=None):
    
    write_step=1
    
    for epoch in range(STARTING_EPOCH,EPOCH+1):
        
        train_loss_value=0
        train_correct_value=0
        train_total_value=0
        
        progress_bar=tqdm.tqdm(range(len(Train_Dataloader)),"Training Progress")
            
        for batch_train,(img,label) in enumerate(Train_Dataloader):
                
            img_train=img.to(devices)
            label_train=label.to(devices)
            
            optimizer.zero_grad()
            CFG_SCALE=torch.randint(1,101,(1,)).item()
            if CFG_SCALE<10:
                labels=None
            t=Diffussion_Model.Random_Timesteps(img_train.shape[0]) # Create Random Tİmesteps
            noisy_img,noise=Diffussion_Model.Noising_to_Image(img_train,t,label_train) # image noising Step
            
            pred_noise=Model(noisy_img,t,label_train) # Pred noise with model VAE
            loss_train=loss_fn(pred_noise,noise) # Loss MSE between pred_noise and real noise
            
            train_loss_value+=loss_train.item()
                        
            loss_train.backward()
            optimizer.step()
            progress_bar.update(1)
            
            if batch_train % 30 == 0 and batch_train>0:
                Tensorboard.add_scalar("Loss_Train",train_loss_value/(batch_train+1),global_step=write_step)
                Tensorboard.add_scalar("Accuracy_Train",((train_correct_value/train_total_value)*100),global_step=write_step)
                write_step+=1    
                progress_bar.set_postfix({"EPOCH":epoch,
                                          "Batch":batch_train+1,
                                          "Loss_Train":train_loss_value/(batch_train+1),
                                          "Accuracy_Train":((train_correct_value/train_total_value)*100)})   
        
        progress_bar.close()    
        
        img_pred=Diffussion_Model.Denoising(Model,img_train.shape[0],labels) 
        Tensorboard.add_image("Predicted Images",img_pred[0],global_step=epoch)     

        Save_Checkpoint_fn(epoch=epoch,optimizer=optimizer,model=Model,save_dir=Checkpoints_dir)