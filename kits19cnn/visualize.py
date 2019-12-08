import matplotlib.pyplot as plt
import os

from typing import Dict, List, Optional, Union  # isort:skip
from collections import defaultdict
from pathlib import Path

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

from catalyst.utils.tensorboard import SummaryItem, SummaryReader

print("If you're using a notebook, "
      "make sure to run %matplotlib inline beforehand.")

def plot_scan(scan, start_with, show_every, rows=3, cols=3):
    """
    Plots multiple scans throughout your medical image.
    Args:
        scan: numpy array with shape (x,y,z)
        start_with: slice to start with
        show_every: size of the step between each slice iteration
        rows: rows of plot
        cols: cols of plot
    Returns:
        a plot of multiple scans from the same image
    """
    fig,ax = plt.subplots(rows, cols, figsize=[3*cols,3*rows])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols), int(i%cols)].set_title("slice %d" % ind)
        ax[int(i/cols), int(i%cols)].axis("off")

        ax[int(i/cols), int(i%cols)].imshow(scan[ind], cmap="gray")
    plt.show()

def plot_scan_and_mask(scan, mask, start_with, show_every, rows=3, cols=3):
    """
    Plots multiple scans with the mask overlay throughout your medical image.
    Args:
        scan: numpy array with shape (x,y,z)
        start_with: slice to start with
        show_every: size of the step between each slice iteration
        rows: rows of plot
        cols: cols of plot
    Returns:
        a plot of multiple scans from the same image
    """
    fig,ax = plt.subplots(rows, cols, figsize=[4*cols, 4*rows])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/cols), int(i%cols)].set_title("slice %d" % ind)
        ax[int(i/cols), int(i%cols)].axis("off")

        ax[int(i/cols), int(i%cols)].imshow(scan[ind], cmap="gray")
        ax[int(i/cols), int(i%cols)].imshow(mask[ind], cmap="jet", alpha=0.5)
    plt.show()

# FROM: https://github.com/catalyst-team/catalyst/blob/master/catalyst/dl/utils/visualization.py
def _get_tensorboard_scalars(
    logdir: Union[str, Path], metrics: Optional[List[str]], step: str
) -> Dict[str, List]:
    summary_reader = SummaryReader(logdir, types=["scalar"])

    items = defaultdict(list)
    for item in summary_reader:
        if step in item.tag and (
            metrics is None or any(m in item.tag for m in metrics)
        ):
            items[item.tag].append(item)
    return items

def _get_scatter(scalars: List[SummaryItem], name: str) -> go.Scatter:
    xs = [s.step for s in scalars]
    ys = [s.value for s in scalars]
    return go.Scatter(x=xs, y=ys, name=name)

def plot_tensorboard_log(
    logdir: Union[str, Path],
    step: Optional[str] = "batch",
    metrics: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> None:
    init_notebook_mode()
    logdir = Path(logdir)

    logdirs = {
        x.name.replace("_log", ""): x
        for x in logdir.glob("**/*") if x.is_dir() and str(x).endswith("_log")
    }

    scalars_per_loader = {
        key: _get_tensorboard_scalars(inner_logdir, metrics, step)
        for key, inner_logdir in logdirs.items()
    }

    scalars_per_metric = defaultdict(dict)
    for key, value in scalars_per_loader.items():
        for key2, value2 in value.items():
            scalars_per_metric[key2][key] = value2

    figs = []
    for metric_name, metric_logs in scalars_per_metric.items():
        metric_data = []
        for key, value in metric_logs.items():
            try:
                data_ = _get_scatter(value, f"{key}/{metric_name}")
                metric_data.append(data_)
            except:  # noqa: E722
                pass

        layout = go.Layout(
            title=metric_name,
            height=height,
            width=width,
            yaxis=dict(hoverformat=".5f")
        )
        fig = go.Figure(data=metric_data, layout=layout)
        iplot(fig)
        figs.append(fig)
    return figs

def plot_metrics(
    logdir: Union[str, Path],
    step: Optional[str] = "epoch",
    metrics: Optional[List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> None:
    """Plots your learning results.
    Args:
        logdir: the logdir that was specified during training.
        step: 'batch' or 'epoch' - what logs to show: for batches or
            for epochs
        metrics: list of metrics to plot. The loss should be specified as
            'loss', learning rate = '_base/lr' and other metrics should be
            specified as names in metrics dict
            that was specified during training
        height: the height of the whole resulting plot
        width: the width of the whole resulting plot
    """
    assert step in ["batch", "epoch"], \
        f"Step should be either 'batch' or 'epoch', got '{step}'"
    metrics = metrics or ["loss"]
    return plot_tensorboard_log(logdir, step, metrics, height, width)

def save_figs(figs_list, save_dir=None):
    """
    Saves plotly figures. (from plot_metrics)
    """
    if save_dir is None:
        save_dir = os.getcwd()

    for fig in figs_list:
        # takes a metric like train/f1/class_0/epoch to f1_class_0_epoch
        train_metric_name = fig["data"][0]["name"]
        split = train_metric_name.split("/")
        metric_name = "".join([f"{name}_" for name in split
                               if not name in ["train", "val"]])[:-1]
        save_name = os.path.join(save_dir, f"{metric_name}.png")
        fig.write_image(save_name)
        print(f"Saved {save_name}...")
