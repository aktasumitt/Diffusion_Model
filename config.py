# For the train
BATCH_SIZE=100
EPOCH=2
LEARNING_RATE=3e-4

# Paths
TENSORBOARD_DIR="Tensorboard"
CHECKPOINT_DIRD="diffusion_checkpoint.pth.tar"
DATASET_DIR="CIFAR"

# For the Difussion
BETA_START=1e-4
BETA_END=0.02
N_TIMESTEPS=1000

# For the checkpoint loading
LOAD_CHECKPOINT=False

# For prediction
PREDICTION=True
PRED_LABEL="ship"



