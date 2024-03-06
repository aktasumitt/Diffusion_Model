import tqdm
from torchvision.utils import make_grid

def Training(EPOCHS,resume_epoch,Img_dataloader,model,optimizer,diffuser,loss_fn,Save_Checkpoints,CALLBACK_PATH,devices,Tensorboard):

    for epoch in range(resume_epoch,EPOCHS):

        loss_model_value=0.0

        progress_bar=tqdm.tqdm(range(len(Img_dataloader)),"Training Process")

        for batch,(img,labels) in enumerate(Img_dataloader):

            img=img.to(devices)
            labels=labels.to(devices)

            t=diffuser.create_timestep()
            x_t,noise=diffuser.apply_noise_to_img(img,t)
            predicted_noise=model(x_t,t,labels)
            loss=loss_fn(predicted_noise,noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            loss_model_value+=loss.item()

        
        # We Try generate image from noise with this function
        sampled_images=diffuser.sampling(model=model,labels=labels)
        
        # embed to Tensorboard to predicted and real image
        real_grid=make_grid(sampled_images,nrow=10)
        Tensorboard.add_image("Real_Image",real_grid,global_step=epoch+1)
        
        predict_grid=make_grid(sampled_images,nrow=10)
        Tensorboard.add_image("Predcted_Image",predict_grid,global_step=epoch+1)

        
        # Save Checkpoints each epoch
        Save_Checkpoints(model=model,
                        optimizer=optimizer,
                        epoch=epoch+1,
                        save_path=CALLBACK_PATH)

        progress_bar.set_postfix({"Epoch":epoch,
                                  "loss_model":loss_model_value/(batch+1)})

        progress_bar.close()