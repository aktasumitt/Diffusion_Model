import torch

def Save_Checkpoints(optimizer,model,epoch:int,save_path:str):
    callbacks={"Epoch":epoch,
               "Optim_vqgan_State":model.state_dict(),
               "Model_vqgan_State":optimizer.state_dict()
               }
    
    torch.save(callbacks,f=save_path)
    
    print("Checkpoints are saved...")




def Load_Checkpoints(checkpoint,optimizer,model):
    
    optimizer.load_state_dict(checkpoint["Model_vqgan_State"])
    model.load_state_dict(checkpoint["Model_disc_State"])
    
    return checkpoint["Epoch"]
