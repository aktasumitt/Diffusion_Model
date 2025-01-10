import checkpoints,config,dataset,diffussion,model,train,prediction
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


# Devices
devices=("cuda" if torch.cuda.is_available() else "cpu")

# Tensorboard
Tensorboard=SummaryWriter(config.TENSORBOARD_DIR,"Diffussion Model Tensorboard")

# Dataset
train_dataset,test_dataset=dataset.Loading_Dataset(config.DATASET_DIR)
# Dataloader
train_dataloader,test_dataloader=dataset.dataloader(train_dataset,test_dataset,config.BATCH_SIZE)

# Some Variables
CHANNEL_SIZE=train_dataset[0][0].shape[0]
NUM_CLASSES=len(train_dataset.classes)
IMG_SIZE=train_dataset[0][0].shape[1]

# Model
Model=model.Unet(CHANNEL_SIZE,devices,NUM_CLASSES).to(devices)
Diffussion_Model=diffussion.Diffussion(config.BETA_START,config.BETA_END,config.N_TIMESTEPS,IMG_SIZE,devices)

# Optimizer and Loss
optimizer=torch.optim.Adam(params=Model.parameters(),lr=config.LEARNING_RATE)
loss_fn=torch.nn.MSELoss()

# Load Checkpoints
initial_epoch=checkpoints.Load_Checkpoint(config.LOAD_CHECKPOINT,save_path=config.CHECKPOINT_DIRD,optimizer=optimizer,Model=Model)

# Training 
train.Training(EPOCH=config.EPOCH,Train_Dataloader=train_dataloader,Model=Model,Diffussion_Model=Diffussion_Model,optimizer=optimizer,
               loss_fn=loss_fn,Save_Checkpoint_fn=checkpoints.Save_Checkpoints,Checkpoints_dir=config.CHECKPOINT_DIRD,devices=devices,
               STARTING_EPOCH=initial_epoch,Tensorboard=Tensorboard)

# Prediction
prediction.Prediction(config.PREDICTION,label=config.PRED_LABEL,class_to_idx=train_dataset.class_to_idx,Model=Model,
                      Diffussion_Model=Diffussion_Model,devices=devices)









