import matplotlib.pyplot as plt

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
