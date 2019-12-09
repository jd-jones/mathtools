import logging
import time
import copy
import collections
import os

import torch
import numpy as np
from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)


# -=( MISC )=------------------------------------------------------------------
def isscalar(x):
    return x.shape == torch.Size([])


def isreal(x):
    """ Check if a pytorch tensor has real-valued type.

    This function is used to extend the classes defined in mfst.semirings to
    wrap pytorch variables.

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    is_real : bool
    """

    real_valued_types = (
        torch.FloatTensor, torch.DoubleTensor, torch.HalfTensor,
        torch.CharTensor, torch.ShortTensor, torch.IntTensor, torch.LongTensor,
        torch.cuda.FloatTensor, torch.cuda.DoubleTensor, torch.cuda.HalfTensor,
        torch.cuda.CharTensor, torch.cuda.ShortTensor,
        torch.cuda.IntTensor, torch.cuda.LongTensor,
    )

    return isinstance(x, real_valued_types)


def selectDevice(gpu_dev_id):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

    if gpu_dev_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_dev_id

    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus > 0:
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    logger.info(f'{num_visible_gpus} GPU(s) visible to script.')
    logger.info(f'Selected device: {device_name}')

    return device


def makeBatch(arrays, **tensor_kwargs):
    batch = torch.stack(tuple(torch.tensor(a, **tensor_kwargs) for a in arrays))
    return batch


# -=( TRAINING & EVALUATION )=-------------------------------------------------
def predictBatch(model, batch_input, device=None, as_numpy=False):
    """ Use a model to make predictions from a minibatch of data.

    Parameters
    ----------
    model : torch.model
        Model to use when predicting from input. This model should return a
        torch array of scores when called, e.g. ``scores = model(input)``.
    batch_input : torch.Tensor, shape (batch_size, seq_length)
        Input to model.
    device : torch.device, optional
        Device to use when processing data with the model.
    as_numpy : bool, optional
        If True, return a numpy.array instead of torch.Tensor. False by default.

    Returns
    -------
    preds : torch.Tensor or numpy.array of int, shape (batch_size, seq_length, num_model_states)
        Model predictions. Each entry is the index of the model output with the
        highest activation.
    outputs : torch.Tensor or numpy.array of float, shape (batch_size, seq_length)
        Model outputs. Each entry represents the model's activation for a
        particular hypothesis. Higher numbers are better.
    """

    if hasattr(model, 'predict'):
        predict = model.predict
    else:
        def predict(outputs):
            __, preds = torch.max(outputs, 1)
            return preds

    batch_input = batch_input.to(device=device)
    outputs = model(batch_input)
    preds = predict(outputs)

    if as_numpy:
        try:
            preds = preds.cpu().numpy()
        except AttributeError:
            preds = tuple(p.cpu().numpy() for p in preds)
        outputs = outputs.cpu().detach().numpy()

    return preds, outputs


def predictSamples(
        model, data_loader,
        criterion=None, optimizer=None, scheduler=None, data_labeled=False,
        update_model=False, device=None, update_interval=None,
        num_minibatches=None, metrics=None, return_io_history=False):
    """ Use a model to predict samples from a dataset; can also update model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to use when making predictions. If `criterion` is not None, the
        parameters of this model are also updated using back-propagation.
    data_loader : torch.utils.data.DataLoader
    criterion : torch.nn.Modules._Loss, optional
        Loss function to use when training or evaluating `model`. If
        `update_model` is True, the loss function is used to update the model.
        Otherwise the loss function is only used to evaluate the model.
    optimizer : torch.optim.Optimizer, optional
        This argument is ignored if `train_model` is False.
    scheduler : torch.optim.LR_scheduler._LRscheduler, optional
    data_labeled : bool, optional
        If True, a sample from ``data_loader`` is the tuple ``data, labels``.
        If False, a sample from ``data_loader`` is just ``data``.
        Default is False.
    update_model : bool, optional
        If False, this function only makes predictions and does not update
        parameters.
    device : torch.device, optional
        Device to perform computations on. Useful if your machine  has GPUs.
        Default is CPU.
    update_interval : int, optional
        This functions logs progress updates every `update_interval` minibatches.
        If `update_interval` is None, no updates will be logged. Default is None.
    num_minibatches : int, optional
        Maximum number of minibatches to evaluate. If `num_minibatches` is None,
        this function iterates though all the samples in `data_loader`.
    metrics : iterable( metrics.RationalPerformanceMetric ), optional
        Performance metrics to use when evaluating the model. Examples include
        accuracy, precision, recall, and F-measure.
    return_io_history : bool, optional
        If True, this function returns a list summarizing the model's input-output
        history. See return value `io_history` for further documentation.

    Returns
    -------
    io_history : iterable( (torch.Tensor, torch.Tensor, torch.Tensor) )
        `None` if ``return_io_history == False``.
    """

    if device is None:
        device = torch.device('cpu')

    if metrics is None:
        metrics = {}
    for m in metrics.values():
        m.initializeCounts()

    if update_model:
        scheduler.step()
        model.train()
    else:
        model.eval()

    io_history = None
    if return_io_history:
        io_history = []

    with torch.set_grad_enabled(update_model):
        for i, sample in enumerate(data_loader):
            if num_minibatches is not None and i > num_minibatches:
                break

            if update_interval is not None and i % update_interval == 0:
                logger.info(f'Predicted {i} minibatches...')

            if data_labeled:
                inputs, labels = sample
            else:
                inputs = sample

            preds, scores = predictBatch(model, inputs, device=device)

            if criterion is not None:
                labels = labels.to(device=device)
                loss = criterion(scores, labels)
                if update_model:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss = loss.item()
            else:
                loss = 0

            if return_io_history:
                batch_io = (preds,) + tuple(sample)
                io_history.append(batch_io)

            # FIXME: This will only count the first sequence in a minibatch
            if metrics:
                try:
                    preds = preds.cpu().numpy().squeeze()
                except AttributeError:
                    preds = tuple(p.cpu().numpy() for p in preds)[0]
                labels = labels.cpu().numpy().squeeze()

            for key, value in metrics.items():
                metrics[key].accumulate(preds, labels, loss)

    return io_history


def trainModel(
        model, criterion, optimizer, scheduler, train_loader, val_loader=None,
        num_epochs=25, train_epoch_log=None, val_epoch_log=None, device=None,
        update_interval=None, metrics=None, test_metric=None, improvement_thresh=0):
    """ Train a PyTorch model.

    Parameters
    ----------
    model :
    criterion :
    optimizer :
    scheduler :
    train_loader :
    val_loader :
    num_epochs :
    train_epoch_log :
    val_epoch_log :
    device :
    update_interval :
    metrics :
    test_metric :

    Returns
    -------
    model :
    last_model_wts :
    """

    logger.info('TRAINING NN MODEL')

    if metrics is None:
        metrics = {}

    if train_epoch_log is None:
        train_epoch_log = collections.defaultdict(list)
    if val_epoch_log is None:
        val_epoch_log = collections.defaultdict(list)

    if device is None:
        device = torch.device('cpu')
    model = model.to(device=device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_metric = 0.0

    init_time = time.time()
    for epoch in range(num_epochs):
        logger.info('EPOCH {}/{}'.format(epoch + 1, num_epochs))

        metrics = copy.deepcopy(metrics)
        _ = predictSamples(
            model, train_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler,
            device=device, metrics=metrics, data_labeled=True,
            update_model=True
        )

        for metric_name, metric in metrics.items():
            train_epoch_log[metric_name].append(metric.evaluate())

        metric_str = '  '.join(str(m) for m in metrics.values())
        logger.info('[TRN]  ' + metric_str)

        if val_loader is not None:
            metrics = copy.deepcopy(metrics)
            _ = predictSamples(
                model, val_loader,
                criterion=criterion, device=device,
                metrics=metrics, data_labeled=True, update_model=False,
            )

            for metric_name, metric in metrics.items():
                val_epoch_log[metric_name].append(metric.evaluate())

            metric_str = '  '.join(str(m) for m in metrics.values())
            logger.info('[VAL]  ' + metric_str)

        if test_metric is not None:
            test_val = metrics[test_metric].evaluate()
            improvement = test_val - best_metric
            if improvement > 0:
                best_metric = test_val
                best_model_wts = copy.deepcopy(model.state_dict())
                improvement_str = f'improved {test_metric} by {improvement:.5f}'
                logger.info(f'Updated best model: {improvement_str}')

    time_str = makeTimeString(time.time() - init_time)
    logger.info(time_str)

    last_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)

    return model, last_model_wts


def plotEpochLog(epoch_log, subfig_size=None, title='', fn=None):
    if subfig_size is None:
        subfig_size = (12, 3)

    num_plots = len(epoch_log)

    figsize = subfig_size[0], subfig_size[1] * num_plots
    fig, axes = plt.subplots(num_plots, figsize=figsize, sharex=True)

    for i, (name, val) in enumerate(epoch_log.items()):
        axis = axes[i]
        axis.plot(val, '-o')
        axis.set_ylabel(name)
    axes[0].set_title(title)
    axes[-1].set_xlabel('Epoch index')
    plt.tight_layout()

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()


def makeTimeString(time_elapsed):
    mins_elapsed = time_elapsed // 60
    secs_elapsed = time_elapsed % 60
    time_str = f'Training complete in {mins_elapsed:.0f}m {secs_elapsed:.0f}s'
    return time_str


# -=( DATASETS )=--------------------------------------------------------------
class ArrayDataset(torch.utils.data.Dataset):
    """ A dataset wrapping numpy arrays stored in memory.

    Attributes
    ----------
    _data : torch.Tensor, shape (num_samples, num_dims)
    _labels : torch.Tensor, shape (num_samples,)
    _device : torch.Device
    """

    def __init__(self, data, labels, device=None, labels_type=None):
        self.num_obsv_dims = data.shape[1]

        if len(labels.shape) == 2:
            self.num_label_types = labels.shape[1]
        elif len(labels.shape) == 1:
            self.num_label_types = np.unique(labels).shape[0]
        else:
            err_str = f"Labels have a weird shape: {labels.shape}"
            raise ValueError(err_str)

        self._device = device

        data = torch.tensor(data, dtype=torch.float)
        labels = torch.tensor(labels)

        if self._device is not None:
            data = data.to(device=self._device)
            labels = labels.to(device=self._device)

        if labels_type == 'float':
            labels = labels.float()

        self._data = data
        self._labels = labels

        logger.info('Initialized ArrayDataset.')
        logger.info(
            f"Data has dimension {self.num_obsv_dims}; "
            f"{self.num_label_types} unique labels"
        )

    def __len__(self):
        return self._data.shape[0]

    def __getitem__(self, i):
        data = self._data[i]
        label = self._labels[i]

        return data, label


# -=( MODELS )=----------------------------------------------------------------
class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_set_size):
        super().__init__()
        self.input_dim = input_dim
        self.out_set_size = out_set_size
        self.linear = torch.nn.Linear(self.input_dim, self.out_set_size)
        logger.info(
            f'Initialized linear classifier. '
            f'Input dim: {self.input_dim}, Output dim: {self.out_set_size}'
        )

    def forward(self, input_seq):
        output_seq = self.linear(input_seq)
        return output_seq


def conv2dOutputShape(
        shape_in, out_channels, kernel_size,
        stride=None, padding=0, dilation=1):

    if stride is None:
        stride = kernel_size

    shape_in = np.array(shape_in[0:2])
    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    padding = np.array(padding)
    dilation = np.array(dilation)

    shape_out = (shape_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    shape_out = np.floor(shape_out).astype(int)

    return shape_out.tolist() + [out_channels]
