import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
import random
import numpy as np




def save_to_tensorboard_sat(rank, logger, file_grp, step, img, title=None, vmin=80, vmax=350, cmap=plt.cm.jet):
    """
    Save satellite images to TensorBoard.
    
    img: np.array or torch.Tensor, shape (H, W) or (H, W, C)
    """
    if rank == 0:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            
            # Determine figure size to match original image resolution
            height, width = img.shape
            dpi = 100
            figsize = width / float(dpi), height / float(dpi)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Save figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=dpi)
            plt.close(fig)
            buf.seek(0)
            
            # Load buffer with PIL and convert to RGB
            pil_img = Image.open(buf).convert("RGB")
            tensor_img = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            
            # Log to TensorBoard
            logger.experiment.add_image(file_grp, tensor_img, global_step=step)

    plt.close('all')  
    
    
def save_2D_tensor_image_as_png(file_path, img_tensor, vmin=80, vmax=350, cmap=plt.cm.jet, dpi=100):
    """
    Saves a single-channel PyTorch tensor as a PNG file using matplotlib.
    
    Args:
        img_tensor (torch.Tensor): 2D tensor representing an image.
        file_path (str): Output PNG file path.
        cmap (str, optional): Matplotlib colormap.
        vmin, vmax (float, optional): Display range for imshow().
        dpi (int): Resolution of the saved figure.
    """
    # Convert tensor -> numpy
    img = img_tensor.detach().cpu().numpy()

    # Determine figure size from resolution
    height, width = img.shape
    figsize = (width / dpi, height / dpi)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Save directly to file
    plt.savefig(file_path, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



def save_to_tensorboard_sat_cartopy(
    rank,
    logger,
    file_grp,
    step,
    img,
    lon_arr,
    lat_arr,
    vmin=0.1,
    vmax=100,
    dpi=50,
):

    import io
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from PIL import Image

    NUM_COLS = 2  # updated from 4 → 5

    # --------------------------------------------------
    # Normalize image tensor
    # --------------------------------------------------
    if isinstance(img, torch.Tensor):
        img[img <= 0.1] = 0.0
        img = img.detach().cpu().numpy()

    img = np.squeeze(img)

    if img.ndim == 2:
        img = img[np.newaxis]

    total, H, W = img.shape
    k = total // NUM_COLS  # was total // 4

    # --------------------------------------------------
    # Convert lat/lon arrays
    # --------------------------------------------------
    if isinstance(lon_arr, torch.Tensor):
        lon_arr = lon_arr.detach().cpu().numpy()

    if isinstance(lat_arr, torch.Tensor):
        lat_arr = lat_arr.detach().cpu().numpy()

    lon_arr = np.asarray(lon_arr)
    lat_arr = np.asarray(lat_arr)

    # remove channel dimension (5,1,128,128) -> (5,128,128)
    lon_arr = np.squeeze(lon_arr)
    lat_arr = np.squeeze(lat_arr)

    # ensure sample dimension
    if lon_arr.ndim == 2:
        lon_arr = lon_arr[np.newaxis]
        lat_arr = lat_arr[np.newaxis]

    # --------------------------------------------------
    # Figure setup
    # --------------------------------------------------
    col_labels = ["Ground Truth", "Prediction"]  # added 5th label

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("turbo")

    fig, axes = plt.subplots(
        k,
        NUM_COLS,                                        # was 4
        figsize=(2.5 * NUM_COLS, 2.5 * k),              # was 2.5 * 4
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=dpi,
    )

    if k == 1:
        axes = axes[np.newaxis, :]

    # --------------------------------------------------
    # Plot panels
    # --------------------------------------------------
    for row in range(k):

        lon_grid = lon_arr[row].reshape(H, W)
        lat_grid = lat_arr[row].reshape(H, W)

        extent = [
            np.nanmin(lon_grid),
            np.nanmax(lon_grid),
            np.nanmin(lat_grid),
            np.nanmax(lat_grid),
        ]

        for col in range(NUM_COLS):  # was range(4)

            ax = axes[row, col]
            panel = img[col * k + row]

            ax.set_extent(extent, crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f5", zorder=0)
            ax.add_feature(cfeature.LAND, facecolor="#f0ede6", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":", zorder=2)
            ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="#7ab8d9", zorder=2)

            ax.pcolormesh(
                lon_grid,
                lat_grid,
                panel,
                cmap=cmap,
                norm=norm,
                transform=ccrs.PlateCarree(),
                shading="nearest",
                zorder=3,
            )

            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.4,
                color="grey",
                alpha=0.5,
                linestyle="--",
            )

            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 5}
            gl.ylabel_style = {"size": 5}

            if row == 0:
                ax.set_title(col_labels[col], fontsize=8, fontweight="bold", pad=3)

            if col == 0:
                ax.set_ylabel(f"Sample {row}", fontsize=7, labelpad=3)

    # --------------------------------------------------
    # Colorbar
    # --------------------------------------------------
    fig.subplots_adjust(right=0.88)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])

    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
    )

    cb.set_label("Rain rate (mm h⁻¹)", fontsize=8)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(f"Rain rates – step {step}", fontsize=9, y=1.005)

    # --------------------------------------------------
    # Send to TensorBoard
    # --------------------------------------------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    plt.close("all")

    buf.seek(0)

    pil_img = Image.open(buf).convert("RGB")

    tensor_img = (
        torch.from_numpy(np.array(pil_img))
        .permute(2, 0, 1)
        .float() / 255.0
    )

    logger.experiment.add_image(file_grp, tensor_img, global_step=step)





def save_to_tensorboard(logger, file_grp, step, img, title=None):
    """Log an already-composed image tensor (CHW or HW, values in [0, 1]).

    Unlike `save_to_tensorboard_sat` this does no colormapping, so it suits
    torchvision grids of natural images.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float().clamp(0.0, 1.0)
    else:
        img = torch.as_tensor(img).float().clamp(0.0, 1.0)

    if img.ndim == 2:
        img = img.unsqueeze(0)

    logger.experiment.add_image(file_grp, img, global_step=step)
