import sys
from gluonts.torch.model.deepar import DeepAREstimator
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

from const import *
from train import *

config = get_config()
train_dataset, val_dataset, predict_dataset = load_dataset(DATASET_WEEK_PATH)
train_dataloader, val_dataloader, predict_dataloader = get_dataloader(
    train_dataset, val_dataset, predict_dataset, config
)

model = DeepAREstimator(
    context_length=56, num_batches_per_epoch=63, hidden_size=40,
    batch_size=51963, freq='W', prediction_length=12,
    num_layers=4, trainer_kwargs={'accelerator': 'gpu', 'max_epochs':100},
)

predictor = model.train(train_dataset, num_workers=0)