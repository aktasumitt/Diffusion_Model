import torch


def Save_Checkpoints(epoch,optimizer,model,save_path:str):
    print("Checkpoint is saving...\n")
    checkpoint={"Epoch":epoch,
                "Optimizer_state":optimizer.state_dict(),
                "Model_state":model.state_dict()}
    
    torch.save(checkpoint,save_path)
    
    

def Load_Checkpoint(Load:bool,save_path,optimizer,Model):
    
    if Load==True:
        checkpoint=torch.load(save_path)
    
        initial_epoch=checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["Optimizer_state"])
        Model.load_state_dict(checkpoint["Model_state"])

        print(f"Checkpoint was loaded,training is starting {initial_epoch+1}.epoch\n")
        
    else: 
        initial_epoch=1
        print("Checkpoint was not loaded,training is starting from scratch\n")
        
    return initial_epoch




    