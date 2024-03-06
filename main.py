import torch
import checkpoints
import dataset
import training
import config
import warnings
import U_Net,Diffussion
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")


# Control Cuda
devices = ("cuda" if torch.cuda.is_available() else "cpu")


# Tensorboard
Tensorboard=SummaryWriter(config.TENSORBOARD_PATH)

# Create Dataset
cifar_data = dataset.DATASET(img_size=config.IMG_SIZE)
cifar_dataloader = dataset.Create_Dataloader(dataset=cifar_data, batch_size=config.BATCH_SIZE)


# Models
diffusion=Diffussion.Diffusion(batch_size=config.BATCH_SIZE,img_size=config.IMG_SIZE,devices=devices)
model=U_Net.U_Net(channels_size=config.CHANNEL_IMG,batch_size=config.BATCH_SIZE,embedding_dim=config.EMBEDDING_DIM,label_size=config.LABEL_SIZE,devices=devices).to(devices)


# Optimizers
optimizer=torch.optim.Adam(params=model.parameters())


# Loss
loss_fn = torch.nn.MSELoss()


# Load Checkpoınt if you want
if config.LOAD_CHECKPOINTS == True:
    checkpoint = torch.load(f=config.CALLBACK_PATH)
    resume_epoch = checkpoints.Load_Checkpoints(checkpoint=checkpoint,
                                                optimizer=optimizer,
                                                model=model)

    print(f"Training is going to start from {resume_epoch}.epoch... ")
    
else:
    resume_epoch = 0
    print("Training is going to start from scratch...")



# Training
training.Training(EPOCHS=config.EPOCHS,
                  resume_epoch=resume_epoch,
                  Img_dataloader=cifar_dataloader,
                  Tensorboard=Tensorboard,
                  model=model,
                  optimizer=optimizer,
                  diffuser=diffusion,
                  loss_fn=loss_fn,
                  Save_Checkpoints=checkpoints.Save_Checkpoints,
                  CALLBACK_PATH=config.CALLBACK_PATH,
                  devices=devices)
