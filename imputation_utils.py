import pandas as pd
from pandasql import sqldf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm
from pytz import timezone
import json
import os
from typing import Optional, Any
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


# FIXED PARAMETERS
VARIABLES = ['ActivityCounts', 'Barometer', 'BloodPerfusion',
             'BloodPulseWave', 'EnergyExpenditure', 'GalvanicSkinResponse', 'HR',
             'HRV', 'RESP', 'Steps', 'SkinTemperature', 'ActivityClass']
SAMPLING_RATE = 1/60
IMAGE_HEIGHT = 370 # height size of standard spectrogram (matplotlib)
NFFT = 255 # 1 window in spectrogram is derived from NFFT datapoints
NOVERLAP = 128 # window hop in spectrogram


# UTILS (for development, visualization etc.)
def get_time_series(dat, variable, day):
    entry = dat[variable][day]
    time_series = np.array([value if value != 'None' else None for value in entry.split(',')], dtype=float) \
        if not isinstance(entry, np.ndarray) else entry # entry could be ndarray or string
    return time_series

def plotter(day, dat):
    """
    Plots time-series of physiological variables for specific day
    """
    plt.figure()
    plt.subplots_adjust(left=0.1,
                        bottom=0.01,
                        right=1.2,
                        top=1.5,
                        wspace=0.4,
                        hspace=0.4)
    length = len(get_time_series(dat, 'HR', day))

    for i, variable in enumerate(VARIABLES):
        time_series = get_time_series(dat, variable, day)

        assert len(time_series) == length, 'time-series data lengths differ!'

        col = 'green' if dat['VAS'][day] == 0.0 else 'red'
        plt.subplot(3, 4, i+1)
        plt.title(variable)
        plt.plot(time_series, col)
        plt.xlim([0, length])

def plotter_spec(day, dat, NFFT=256, noverlap=128):
    """
    Plots spectrograms of physiological variables for specific day
    """
    plt.figure()
    plt.subplots_adjust(left=0.1,
                        bottom=0.01,
                        right=1.2,
                        top=1.5,
                        wspace=0.4,
                        hspace=0.4)
    length = len(get_time_series(dat, 'HR', day))

    for i, variable in enumerate(VARIABLES):
        time_series = get_time_series(dat, variable, day)

        assert len(time_series) == length, 'time-series data lengths differ!'

        plt.subplot(3, 4, i+1)
        plt.title(variable)
        plt.specgram(time_series, Fs=SAMPLING_RATE, NFFT=NFFT, noverlap=NOVERLAP)

def na_visualizer(days, dat):
    """
    Plots missing data for specific days in red/green
    """
    for day in days:
        length = get_time_series(dat, 'ActivityCounts', day).shape[0]
        n_variables = len(VARIABLES)
        na_matrix = np.zeros((n_variables, length))

        for i, variable in enumerate(VARIABLES):
            time_series = get_time_series(dat, variable, day)

            na_data = np.where(np.isnan(time_series), 0.0, 1.0) # NaN -> 0, data -> 1
            na_matrix[i, :] = na_data

        # hack: below makes sure colors in image are matched correctly (if all missing/no missing would break otherwise)
        na_matrix[1, 1] = 0.0
        na_matrix[2, 1] = 1.0

        # plot
        cmap = matplotlib.colors.ListedColormap(['red', 'green'])
        plt.figure()
        plt.imshow(na_matrix, cmap=cmap, aspect='20', interpolation='nearest')
        plt.title(f'day {day}')
        plt.xlabel('data length [min]')
        plt.ylabel('variable')

def na_sequence_lengths(dat):
    """
    Calculates lengths of missing data sequences of full dataset
    """
    # check NA sequence lengths
    n_days, n_cols = dat.shape

    total_missing_values = 0
    na_sequences_data = []
    for day in range(n_days):
        for variable in VARIABLES:
            time_series = get_time_series(dat, variable, day)

            na_sequences = []
            na_seq = 0
            for datapoint in time_series:
                if not np.isnan(datapoint):
                    if na_seq > 0:
                        na_sequences.append(na_seq)
                        total_missing_values += na_seq
                        na_seq = 0
                else:
                    # datapoint is NA
                    na_seq += 1
            na_sequences_data += na_sequences

    # show numerical distribution of NA sequence lengths
    temp = pd.DataFrame(np.array(na_sequences_data), columns=['n'])
    query = '''
    SELECT n AS sequence_length, COUNT(*) AS occurrences
    FROM temp
    GROUP BY n;'''
    temp = sqldf(query)

    # plot distribution of NA sequence lengths
    temp.plot.bar(x='sequence_length', y='occurrences', width=2, title=f'Missing data (total: {total_missing_values})')
    plt.xticks([])
    plt.xlim([-10, temp.shape[0]])

    return temp

def missing_data_per_variable(dat):
    """
    Calculates missing data ratio for each variable
    """
    dat2 = dat.copy()

    missing_data_variable = {variable: [] for variable in VARIABLES}
    for variable in VARIABLES:
        for day in range(dat2.shape[0]):
            time_series = get_time_series(dat, variable, day)
            na_data = list(np.where(np.isnan(time_series), 0.0, 1.0)) # NaN -> 0, data -> 1
            missing_data_variable[variable] += na_data

    missing_data_variable = {variable: np.mean(na_data) for variable, na_data in missing_data_variable.items()}
    return missing_data_variable

def imputer(dat, method, **kwargs):
    """Imputes data by day (!) according to specified method"""
    data_imputed = dat.copy()

    assert method in ('mean', 'median', 'mode', 'linear', 'quadratic', 'spline', 'nearest')
    if method == 'mean':
        value = data_imputed.mean(axis=1)
        data_imputed = data_imputed.transpose().fillna(value).transpose() # not very pretty but works
    elif method == 'median':
        value = data_imputed.median(axis=1)
        data_imputed = data_imputed.transpose().fillna(value).transpose() # not very pretty but works
    elif method == 'mode':
        value = data_imputed.mode(axis=1)[0]
        data_imputed = data_imputed.transpose().fillna(value).transpose() # not very pretty but works
    else:
        MAX_FILL = 1440 # maximum imputation window (from both sides!)
        data_imputed = data_imputed.interpolate(method=method, axis=1, limit=MAX_FILL, limit_direction='both', fill_value='extrapolate', **kwargs)

    return data_imputed

def visualize_imputation(dat, method, **kwargs):
    """Plots data vs. imputed data"""
    data_imputed = imputer(dat, method, **kwargs)

    # visualize imputation
    plt.subplots_adjust(left=0.1,
                        bottom=0.01,
                        right=1.7,
                        top=1.0,
                        wspace=0.4,
                        hspace=0.4)

    for i, variable in enumerate(VARIABLES):
        plt.subplot(2, 5, i+1)
        plt.title(variable)

        # post-imputation
        ax = data_imputed.iloc[i].plot(color='red', linewidth=1.0)
        ax.set_xticklabels([], minor=True)

        # pre-imputation
        ax = dat.iloc[i].plot()
        ax.set_xticklabels([], minor=True)


# DATA LOADING/PROC UTILS
def import_data(discard_variables=True, discard_days=True, THRESHOLD=60):
    """Imports data in single dataframe"""
    # file path to data folder
    path = './Output'

    # import
    file = path + f'/combined_data.csv'
    data = pd.read_csv(file, index_col=0).fillna(pd.NA)

    # discard variables from dataset (too much missing data)
    if discard_variables:
        to_discard = ['GalvanicSkinResponse', 'ActivityClass']

        data = data.drop(columns=to_discard)
        global VARIABLES
        VARIABLES = [variable for variable in VARIABLES if variable not in to_discard]

        print(f'discarded variables: {to_discard}')

    # discard days with few data
    if discard_days:
        discarded_days = []

        # days with less than THRESHOLD [min] of data
        days = range(data.shape[0])
        for day in days:
            time_series = get_time_series(data, VARIABLES[0], day)
            length = len(time_series)

            # if cannot build at least one segment -> discard day
            if length < THRESHOLD:
                discarded_days.append(day)

        print(f'discarded days (less than {THRESHOLD}min of data): {discarded_days}')
        temp = [] # just for print information

        # days where sensor is out for full day
        for day in days:
            for variable in VARIABLES:
                time_series = get_time_series(data, variable, day)

                # if only missing data for full day -> discard day
                if np.sum(np.where(np.isnan(time_series), 0.0, 1.0)) == 0: # NaN -> 0, data -> 1
                    discarded_days.append(day)
                    temp.append(day)

        data = data.drop(discarded_days)
        data = data.reset_index(drop=True)

        print(f'discarded days (sensor out all day): {temp}')

    return data, VARIABLES

def data_to_days(dat) -> list:
    """
    Returns a list of daily data (needed for imputation)
    :param dat: full data
    :return: list of daily data
    """
    data_daily = []

    n_days, _ = dat.shape
    for day in tqdm(range(n_days)):
        # create dataframe with data by day (i.e. each row is full daily data of one variable, each column is a one-minute measurement)

        # timestamps
        date = dat["date"].iloc[day] # date (year, month, day) of current day
        timestamps_full = pd.to_datetime(
            [f'{str(hour)}:{str(minute)}, {date}' for hour in range(0, 24)
             for minute in range(0, 60)]) # timestamps for one full day
        timestamps_available = pd.to_datetime(
            [f'{ts}, {date}' for ts in dat['Timestamps'].iloc[day].split(',')]) # timestamps with data

        # assign data to correct timestamp
        rows = [{timestamp: np.NaN for timestamp in timestamps_full} for variable in VARIABLES] # initialize full day with NaNs
        rows_available = [get_time_series(dat, variable, day) for variable in VARIABLES] # available data for day
        for variable, _ in enumerate(VARIABLES):
            for i, timestamp in enumerate(timestamps_available):
                rows[variable][timestamp] = rows_available[variable][i] # fill timestamps where we have data

        # build dataframe
        row_names = VARIABLES
        column_names = timestamps_full
        data_day = pd.DataFrame(data=rows,
                                index=row_names,
                                columns=column_names)

        data_daily.append(data_day)

    return data_daily

def normalize_daily_variables(data_day):
    """Normalized each variable in data of one
    day"""
    n_rows, n_cols = data_day.shape
    for row in range(n_rows):
        data_day.iloc[row, :] = StandardScaler().fit_transform(np.array([data_day.iloc[row, :]]).reshape(-1, 1)).reshape(-1)

def normalize_by_day(dat, check=True):
    """Normalizes each day variable-wise"""
    for data_day in dat:
        normalize_daily_variables(data_day)

    # print out mean/std
    if check:
        for day, data_day in enumerate(dat):
            n_rows, n_cols = data_day.shape
            for row in range(n_rows):
                print(f'day {day} - {VARIABLES[row]}: '
                      f'mean: {np.mean(data_day.iloc[row, :])}, '
                      f'std: {np.std(data_day.iloc[row, :])}')


# IMPUTATION UTILS
def masker(dat, lm=3, masking_ratio=0.15):
    """
    Masks sequences (set to NaN) for training/testing
    :param dat: data by day (!)
    :param lm: mean sequence length
    :param masking_ratio: ratio of masking/non-masking
    :return: mask of same shape as data with (0: available, 1: masked (purpose-fully set to NaN), 2: missing (NaN from beginning))
    """
    # full mask is for full day, mask is for one variable of the day
    full_mask = np.zeros(dat.shape) # whether datapoint is either: 0: available, 1: masked (purpose-fully set to NaN), 2: missing (NaN from beginning)

    for variable in range(dat.shape[0]):
        row = dat.iloc[variable]

        # init mask
        mask = np.zeros(1440) # whether datapoint is either: 0: available, 1: masked (purpose-fully set to NaN), 2: missing (NaN from beginning)
        mask[np.isnan(row)] = 2 # set already missing data

        p_start_masking_seq = masking_ratio
        p_start_keep_seq = 1 - p_start_masking_seq
        p_terminate_masking_seq = 1 / lm
        p_terminate_keep_seq = p_terminate_masking_seq * masking_ratio / (1 - masking_ratio)

        data_missing = False # whether data is already missing (not purposefully masked)
        masking = (np.random.rand() < p_start_masking_seq) # True: we are in masking sequence, False: we are in keep sequence
        for i in range(len(row)):
            # check if datapoint is already missing
            data_missing = (mask[i] == 2)

            if not data_missing:
                # check if we just exited from already missing data sequence -> in that case start again in masking state with prob
                try:
                    if mask[i - 1] == 2:
                        masking = (np.random.rand() < p_start_masking_seq) # True: we are in masking sequence, False: we are in keep sequence
                except IndexError:
                    pass

                # assign masking or not
                mask[i] = int(masking) # 0: available, 1: masked (purpose-fully set to NaN), 2: missing (NaN from beginning)

                # check if we terminate sequence
                if np.random.rand() < {True: p_terminate_masking_seq, False: p_terminate_keep_seq}[masking]:
                    masking = not masking # masking -> keep, keep -> masking

        full_mask[variable, :] = mask

    return full_mask

def visualize_mask(mask, day='?'):
    """
    Visualizes masked data (green: available data, red: already missing data, blue: masked data)
    :param mask: mask with same shape as data by day (!)
    """
    length = mask.shape[0]

    # hack: below makes sure colors in image are matched correctly (if all missing/no missing would break otherwise)
    mask_copy = mask.copy()
    mask_copy[0, 0] = 0.0
    mask_copy[1, 0] = 1.0
    mask_copy[2, 0] = 2.0

    # plot
    cmap = matplotlib.colors.ListedColormap(['green', 'blue', 'red'])
    plt.figure()
    plt.imshow(mask_copy, cmap=cmap, aspect='20', interpolation='nearest')
    plt.title(f'day {day}')
    plt.xlabel('data length [min]')
    plt.ylabel('variable')
    #plt.colorbar()
    plt.show()

def test_imputation_methods(dat: list, lm=3, masking_ratio=0.15) -> dict:
    """
    Tests all imputation methods
    :param dat: data by day (!)
    :param lm: mean sequence length
    :param masking_ratio: ratio of masking/non-masking
    :return: mean absolute error (MAE) & mean relative error (MRE) on masked data (sorted dict)
    """
    n_days = len(dat)
    imputation_methods = ('mean', 'median', 'mode', 'linear', 'quadratic', 'spline', 'nearest')

    # build masks
    mask_shape = dat[0].shape
    masks = np.zeros((n_days, *mask_shape))
    for day in range(n_days):
        masks[day] = masker(dat[day], lm=lm, masking_ratio=masking_ratio)

    # score each imputation method
    scores = {}
    for imputation_method in imputation_methods:
        imputation_erorrs = np.array([])
        reals = np.array([]) # for MRE
        for day in range(n_days):
            data_day = dat[day] # data for current day
            mask = masks[day] # mask for current day (all imputation methods use same masks)

            # impute masked data
            masked_data = pd.DataFrame(np.where(mask == 1.0, np.NaN, data_day)) # masked -> NaN
            data_imputed = imputer(masked_data, imputation_method, order=2) # order for spline

            # calculate error
            real_data = data_day.to_numpy()[mask == 1.0]
            imputed_data = data_imputed.to_numpy()[mask == 1.0]

            # save
            imputation_errors = np.concatenate((imputation_erorrs,
                                                np.abs(real_data - imputed_data)),
                                               axis=None)
            reals = np.concatenate((reals, real_data), axis=None)

        mae = np.mean(imputation_errors) # MAE
        mre = np.sum(imputation_errors) / np.sum(np.abs(reals)) # MRE
        scores[imputation_method] = (mae, mre)


    return sorted(scores.items(), key=lambda x: x[1][0]) # sort by MAE


# TRANSFORMER
def _get_activation_fn(activation):
    """
    Choose between ReLU and GELU
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()

        # Sinusoidal positional encoding
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        # Sinusoidal positional encoding
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Linear positional encoding (learnable)
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()

        # Linear positional encoding
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True

        # Initialization
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        """
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        # Linear positional encoding
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    """
    Choose between learnable and sinusoidal encoding
    """
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    """
    One transformer encoder layer
        It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()

        # Self-attention layer
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = Dropout(dropout)
        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps

        # Feedforward layer
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # Self-attention layer
        # a) Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # b) Add + residual dropout
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        # c) For PyTorch compatibility
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # d) Layer normalization
        src = self.norm1(src)

        # Feed-forward network
        # a) For PyTorch compatibility
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        # b) MLP
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # c) Add + residual dropout
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        # d) For PyTorch compatibility
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # e) Layer normalization
        src = self.norm2(src)
        # f) For PyTorch compatibility
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)

        return src


class TSTransformerEncoder(nn.Module):
    """
    Transformer for time-series (transduction/imputation)
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        # Transformer parameters
        self.max_len = max_len # input dimension
        self.d_model = d_model # embedding dimension
        self.n_heads = n_heads # encoder heads
        self.feat_dim = feat_dim # number of variables (sensors) in time-series
        self.dropout1 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, feat_dim) # linear transformation back to input dimension

        # Embedding
        self.project_inp = nn.Linear(feat_dim, d_model)

        # Positional encoding
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        # Type of normalization (BN/LN)
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        # Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # ReLU or GELU in FFN
        self.act = _get_activation_fn(activation)

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # Embedding + PE
        # a) For PyTorch compatibility
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        # b) Embedding of input sequence (dimension: seq_length -> d_model)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        # c) Positional encoding
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer

        # Encoder
        # a) Get embedding through encoder (with paddings (for missing data))
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        # b) Apply activation-function
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        # c) Reshape
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        # d) Final dropout
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        # Linear transformation
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Transformer for time-series (regression/classification)
        Simplest classifier/regressor. Can be either regressor or classifier because the output does not include softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        # Transformer parameters
        self.max_len = max_len # input dimension
        self.d_model = d_model # embedding dimensions
        self.n_heads = n_heads # encoder heads
        self.feat_dim = feat_dim # number of variables (sensors) in time-series
        self.num_classes = num_classes # output dimension
        self.dropout1 = nn.Dropout(dropout)
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

        # Embedding
        self.project_inp = nn.Linear(feat_dim, d_model) # feat_dim: # variables in multivariate time-series

        # Positional encoding
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        # Type of normalization (BN/LN)
        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        # Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # ReLU or GELU in FFN
        self.act = _get_activation_fn(activation)

    def build_output_module(self, d_model, max_len, num_classes):
        """MLP after encoder"""
        # MLP
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed, add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        # Embedding + PE
        # a) For PyTorch compatibility
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        # b) Embedding of input sequence (dimension: seq_length -> d_model)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        # c) Positional encoding
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer

        # Encoder
        # a) Get embedding through encoder (with paddings (for missing data))
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        # b) Apply activation-function
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        # c) Reshape
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        # d) Final dropout
        output = self.dropout1(output)

        # MLP
        # a) Padding -> 0.0
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        # b) Reshape
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        # c) Run through MLP
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output

def l2_reg_loss(model):
    """
    Returns the squared L2 norm of output layer of given model
    """

    for name, param in model.named_parameters():
        if name == 'output_layer.weight':
            return torch.sum(torch.square(param))


class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Custom cross-entropy loss
        pytorch's CrossEntropyLoss is fussy:
        1) needs Long (int64) targets only, and
        2) only 1D.
        This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """
    Masked MSE Loss (MSE on unmasked values)
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        Compute the loss between a target value and a prediction.
        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered
        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)





# TODO: dataloader (one day) + normalization

# TODO: baseline imputation methods

# TODO: tester (+ vs. baselines)

# TODO: visualizatin of imputation

# TODO: full transformer model here to be callable for preproc
#%%
