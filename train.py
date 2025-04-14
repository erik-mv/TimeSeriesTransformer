import logging
from datasets import load_from_disk
import pandas as pd
import pickle
from datetime import datetime
from evaluate import load
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction, PretrainedConfig
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    AddTimeFeatures,
    RemoveFields,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.transform.sampler import InstanceSampler
from typing import Optional, Iterable
import torch
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from accelerate import Accelerator
from torch.optim import AdamW
from gluonts.time_feature import time_features_from_frequency_str

time_features = time_features_from_frequency_str("1W")

from const import *

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
    level=logging.INFO,
    )
logger = logging.getLogger(__name__)

FREQ = "1W"
PREDICTION_LENGTH = 12

LAG_SEQUENCE = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    16, 20, 24, 28, 32, 36, 40, 44, 48,
    50, 51, 52, 53, 54, 55, 56, 
]

LAG_SEQUENCE = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    16, 20, 24, 28, 32, 36, 40, 44, 48,
    50, 51, 52, 53, 54, 55, 56, 
]

NUM_STATIC_REAL_FEATURES = 450 #224
NUM_DYNAMIC_REAL_FEATURES = 20
PARTIAL = 100

NUM_INN_ID = 51963
NUM_IPUL = 3
NUM_ID_REGION = 86
NUM_MAIN_OKVED_GROUP = 87
NUM_PAYER_CAT_FEAT = 28
NUM_PAYEE_CAT_FEAT = 30


def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
    return batch

def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # month of year in the case when freq="M"
            # these serve as positional encodings
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # step 6: vertically stack all the temporal features into the key FEAT_TIME
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                + (
                    [FieldName.FEAT_DYNAMIC_REAL]
                    if config.num_dynamic_real_features > 0
                    else []
                ),
            ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "val", "predict"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "val": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "predict": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_train_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, "train")

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )

def create_val_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data)

    # we create a Validation Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "val")

    # we apply the transformations in train mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=True)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def create_predict_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # We create a test Instance splitter to sample the very last
    # context window from the dataset provided.
    instance_sampler = create_instance_splitter(config, "predict")

    # We apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )

def get_config():
    return TimeSeriesTransformerConfig(
        prediction_length=PREDICTION_LENGTH,
        # context length:
        context_length=56,  #56  PREDICTION_LENGTH*3
        #distribution_output = "student_t"  # "student_t", 'normal' или 'negative_binomial'.
        #scaling = 'mean', # 'std' or  None 'mean'
        # lags coming from helper given the freq:
        lags_sequence=LAG_SEQUENCE,
        # we'll add 2 time features ("month of year" and "age", see further):
        num_time_features=len(time_features) + 1,
        # we have a single static categorical feature, namely time series ID:
        num_static_categorical_features=5,#6,
        num_static_real_features=NUM_STATIC_REAL_FEATURES,
        num_dynamic_real_features=NUM_DYNAMIC_REAL_FEATURES,
        # it has 366 possible values:
        #cardinality=[NUM_INN_ID, NUM_IPUL, NUM_ID_REGION, NUM_MAIN_OKVED_GROUP, NUM_PAYER_CAT_FEAT, NUM_PAYEE_CAT_FEAT],
        cardinality=[NUM_IPUL, NUM_ID_REGION, NUM_MAIN_OKVED_GROUP, NUM_PAYER_CAT_FEAT, NUM_PAYEE_CAT_FEAT],
        # the model will learn an embedding of size 2 for each of the 366 possible values:
        embedding_dimension=[2, 3, 3, 3, 3],
        #embedding_dimension=[12, 2, 4, 4, 4, 4],
        # transformer params:
        encoder_layers=4, # 4 6
        decoder_layers=4, # 4 6
        d_model=64,        # 32  128
    )

@torch.no_grad()
def evaluate(model, model_path, val_dataloader, val_dataset, device, config):
    model.eval()
    forecasts = []
    for _, batch in enumerate(val_dataloader):
        outputs_val = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs_val.sequences.cpu().numpy())
    forecasts_path = model_path.parent / f'forecasts_{model_path.name.split('-')[1].split('=')[-1]}.pkl'
    logger.info('Forecasts path: %s', forecasts_path)
    with open(str(forecasts_path),'wb') as f:
        pickle.dump(forecasts, f)
    forecasts = np.vstack(forecasts)
    forecast_median = np.median(forecasts, 1)
    forecast_median[forecast_median<0.0] = 0.0
    rmsle_pablic = []
    rmsle_privat = []
    for item_id, ts in enumerate(val_dataset):
        ground_truth = ts["target"][-PREDICTION_LENGTH:]
        rmsle_pablic.append(np.sqrt(((ground_truth[:4] - forecast_median[item_id,:4])**2).mean()))
        rmsle_privat.append(np.sqrt(((ground_truth[4:] - forecast_median[item_id,4:])**2).mean()))
    return {
        'mean_value': round((np.expm1(forecast_median[:, :4])).mean()),
        'rmsle_pablic': round(np.array(rmsle_pablic).mean(), 4),
        'rmsle_privat': round(np.array(rmsle_privat).mean(), 4),
    }


@torch.no_grad()
def predict(model, model_path, dataloader, device, config):
    model.eval()
    forecasts = []
    for _, batch in enumerate(dataloader):
        outputs_val = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs_val.sequences.cpu().numpy())
    forecasts_path = model_path.parent / f'forecasts_{model_path.name.split('-')[1].split('=')[-1]}.pkl'
    logger.info('Forecasts path: %s', forecasts_path)
    with open(str(forecasts_path),'wb') as f:
        pickle.dump(forecasts, f)
    forecasts = np.vstack(forecasts)
    #for index in range(10):
    #    forecasts_percentile = [
    #        data[np.delete(np.arange(100), np.unique(np.where((data<np.percentile(data, 10-index, axis=0)) | (data>np.percentile(data, 90+index, axis=0)))[0])),:]
    #        for data in forecasts
    #    ]
    #    len_list = [len(data)for data in forecasts_percentile]
    #    if min(len_list)>10:
    #        logger.info('Percentile = [%d, %d]', 10-index, 90+index)
    #        break
    #forecasts_percentile = np.array([np.expm1(np.median(data, axis=0)) for data in forecasts])
    forecasts_percentile = np.expm1(np.median(forecasts, axis=1))
    forecasts_4 = round(forecasts_percentile[:, :4].mean())
    forecasts_8 = round(forecasts_percentile[:, 4:].mean())
    forecasts = round(forecasts_percentile.mean())
    return forecasts, forecasts_4, forecasts_8

def load_dataset(dataset_week_path, add_val=True):
    dataset = load_from_disk(str(dataset_week_path))
    if add_val:
        train_dataset = dataset["train"]
        val_dataset = dataset["val"]
        predict_dataset = dataset["predict"]
    else:
        train_dataset = dataset["val"]
        val_dataset = dataset["val"]
        predict_dataset = dataset["predict"]
    train_dataset.set_transform(partial(transform_start_field, freq=FREQ))
    val_dataset.set_transform(partial(transform_start_field, freq=FREQ))
    predict_dataset.set_transform(partial(transform_start_field, freq=FREQ))
    return train_dataset, val_dataset, predict_dataset

def get_dataloader(train_dataset, val_dataset, predict_dataset, config):
    train_dataloader = create_train_dataloader(
        config=config,
        freq=FREQ,
        data=train_dataset,
        batch_size=13000,  #51963   13000  51963
        num_batches_per_epoch=83,# 251   83
    )
    val_dataloader = create_val_dataloader(
        config=config,
        freq=FREQ,
        data=val_dataset,
        batch_size=256,
    )
    predict_dataloader = create_val_dataloader(  #create_predict_dataloader
        config=config,
        freq=FREQ,
        data=predict_dataset,
        batch_size=256,
    )
    return train_dataloader, val_dataloader, predict_dataloader

def train(config, train_dataloader, val_dataloader, predict_dataloader, val_dataset, add_val=True):
    logger.info(42 * '#')
    logger.info('Add val = %s', add_val)
    model = TimeSeriesTransformerForPrediction(config)
    accelerator = Accelerator()
    device = 'cuda:0'  #'cpu'
    #device = 'cpu'
    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95))   #betas=(0.9, 0.95), #weight_decay=1e-3
    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )   
    model.to(device)
    score_min = None
    foldername = MODELS_PATH / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    foldername.mkdir(parents=True, exist_ok=True)
    patience = PARTIAL
    for epoch in range(2000):
        loss_mean = []
        model.train()
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss
            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            loss_mean.append(loss.item())
            if idx % 10 == 0:
                logger.info('Epoch: %d,\tBatches_per: %d,\tLoss_train: %0.4f', epoch+1, idx, loss.item())
        loss_mean = round(np.array(loss_mean).mean(), 4)
        torch.cuda.empty_cache()
        model_path = foldername / f'checkpoint-epoch={epoch+1}-train_loss={loss_mean}'
        if add_val:
            score = evaluate(model, model_path, val_dataloader, val_dataset, device, config)
            log_score = f'Epoch: {epoch+1},\tLoos_mean_train: {loss_mean},\t'
            for key, value in score.items():
                log_score += f'{key}: {value}\t'
            logger.info(log_score)
            accelerator.save_state(
                foldername / f'checkpoint-epoch={epoch+1}-train_loss={loss_mean}-rmsle_pablic={score['rmsle_pablic']}-rmsle_privat={score['rmsle_privat']}'
            )
            if score_min is None:
                score_min = score['rmsle_pablic']
            elif score_min < score['rmsle_pablic']:
                patience -= 1
            else:
                score_min = score['rmsle_pablic']
                patience = PARTIAL
        else:
            mean_value, mean_value_4, mean_value_8 = predict(model, model_path, predict_dataloader, device, config)
            model_path = model_path.parent / f'{model_path.name}_mean_value={mean_value}_{mean_value_4}_{mean_value_8}'
            accelerator.save_state(
                model_path
            )
        if patience==0:
            logger.info('Easy stop!!!')
            break
    torch.cuda.empty_cache()
    
def inference(model_path:Path, train_dataloader, dataloafer_inference, config):
    accelerator = Accelerator()
    model = TimeSeriesTransformerForPrediction(config)
    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    accelerator.load_state(model_path)
    device = 'cuda:0'
    model.to(device)
    model.eval()
    forecasts = []
    for _, batch in tqdm(enumerate(dataloafer_inference)):
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts.append(outputs.sequences.cpu().numpy())
    torch.cuda.empty_cache()
    forecasts_path = model_path.parent / f'forecasts_{model_path.name.split('-')[1].split('=')[-1]}.pkl'
    logger.info('Forecasts path: %s', forecasts_path)
    with open(str(forecasts_path),'wb') as f:
        pickle.dump(forecasts, f)

def run_pipeline(add_val):
    """Run pipeline from arg"""
    config = get_config()
    if not (add_val is None):
        add_val = bool(add_val)
        train_dataset, val_dataset, predict_dataset = load_dataset(DATASET_WEEK_PATH, add_val)
        logger.info(
            'Len train: %d\tval: %d\tpredict: %d',
            len(train_dataset[0]['target']),
            len(val_dataset[0]['target']),
            len(predict_dataset[0]['target']),
        )
        train_dataloader, val_dataloader, predict_dataloader = get_dataloader(
            train_dataset, val_dataset, predict_dataset, config,
        )
        train(config, train_dataloader, val_dataloader, predict_dataloader, val_dataset, add_val)

def handle_args(args):
    """
    Extract command line args and call delegate function.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    return run_pipeline(
        args.train,
    )

def main(args=None):
    """
    Parse command line args and call handler when run as a script.
    Parameters
    ----------
    args : list
        Command line arguments as a list of strings [optional]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", type=int,
        help="'1' or '0' for used val data",
    )
    parser.set_defaults(func=handle_args)
    args = parser.parse_args()
    # Call args default handler
    args.func(args)

if __name__ == '__main__':
    main()