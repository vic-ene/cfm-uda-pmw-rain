import sys
import os
from natsort import natsorted
from glob import glob as gglob

import torch
import copy 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.util import add_cyclic_point
import cartopy.util as cutil

import tqdm

import glob


from scipy.spatial import ConvexHull


from shapely.geometry import Point, Polygon
import scipy





MIN_TB = 80
MAX_TB = 350


import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_DPR_rainrate_multi(
    titles,
    lons,
    lats,
    rains,
    extent,
    save_path="",
    vmin=0.1,
    vmax=100,
    lon_trace_override=None,
    lat_trace_override=None,
    save_individual=False,   
    trace_idx=(1, 2),
    c1="grey",
    c2="grey",
    alpha_val=0.1,
):  

    if isinstance(rains[0], torch.Tensor):
        rains = [elt.cpu().numpy() for elt in rains]

    plt.close('all') 
    n = len(rains)

    fig, axes = plt.subplots(
        1, n,
        figsize=(5.5 * n, 5),
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    axes = np.atleast_1d(axes)

    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    

    # Prepare traces
    t_i_0, t_i_1 = trace_idx
    t_i = t_i_0
    lon_trace_0 = np.concatenate([lons[t_i][0,:], lons[t_i][-1,:], lons[t_i][:,0], lons[t_i][:,-1]])
    lat_trace_0 = np.concatenate([lats[t_i][0,:], lats[t_i][-1,:], lats[t_i][:,0], lats[t_i][:,-1]])

    t_i = t_i_1
    lon_trace_1 = np.concatenate([lons[t_i][0,:], lons[t_i][-1,:], lons[t_i][:,0], lons[t_i][:,-1]])
    lat_trace_1 = np.concatenate([lats[t_i][0,:], lats[t_i][-1,:], lats[t_i][:,0], lats[t_i][:,-1]])

    if isinstance(lon_trace_override, np.ndarray):
        lon_trace_1 = np.concatenate([
            lon_trace_override[0,:],
            lon_trace_override[-1,:],
            lon_trace_override[:,0],
            lon_trace_override[:,-1]
        ])

    if isinstance(lat_trace_override, np.ndarray):
        lat_trace_1 = np.concatenate([
            lat_trace_override[0,:],
            lat_trace_override[-1,:],
            lat_trace_override[:,0],
            lat_trace_override[:,-1]
        ])

    size_labels = 12
    fs = 18

    for i, (ax, title, lon, lat, rain) in enumerate(
        zip(axes, titles, lons, lats, rains)
    ):

        if not isinstance(lat_trace_override, np.ndarray):
            ax.set_title(title, weight='bold', fontsize=fs)

        lon = copy.deepcopy(lon)
        lat = copy.deepcopy(lat)
        rain = copy.deepcopy(rain)

        # Gridlines
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1.0,
            color='gray',
            alpha=0.5,
            linestyle='--'
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xlabel_style = {'size': size_labels}
        gl.ylabel_style = {'size': size_labels}

        # Features
        ax.add_feature(cfeature.OCEAN.with_scale('50m'))
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')

        # Trace scatter
        ax.scatter(
            lon_trace_0, lat_trace_0,
            c=c1, s=2,
            transform=ccrs.PlateCarree(),
            alpha=alpha_val,
        )

        ax.scatter(
            lon_trace_1, lat_trace_1,
            c=c2, s=2,
            transform=ccrs.PlateCarree(),
            alpha=alpha_val,
        )

        if len(rain.shape) == 3:
            rain = rain.squeeze(0)

        # Flatten + sort
        lon = lon.ravel()
        lat = lat.ravel()
        rain = rain.ravel()

        idx = np.argsort(rain)

        lon = lon[idx]
        lat = lat[idx]
        rain = rain[idx]

        cmap = copy.deepcopy(mpl.cm.get_cmap("turbo"))
        cmap.set_bad((0, 0, 0, 0))   # NaN -> fully transparent

        # Scatter
        sc = ax.scatter(
            lon, lat, c=rain,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            s=0.3,
            edgecolors="none",
            antialiased=False,
        )

        ax.set_extent(extent, crs=ccrs.PlateCarree())

        cb = fig.colorbar(sc, ax=ax, orientation="vertical")
        cb_ax = cb.ax   # <--- keep reference
        cb.set_label("Rain Rate (mm/h)", fontsize=size_labels)

        # ----------------------------------
        # SAVE INDIVIDUAL SUBPLOT + ALL TEXT
        # ----------------------------------
        if save_individual and save_path:

            fig.canvas.draw()  # force layout so positions are known

            renderer = fig.canvas.get_renderer()

            # Dynamically match colorbar height to plot height
            pos = ax.get_position()
            cb_pos = cb_ax.get_position()
            cb_ax.set_position([
                cb_pos.x0,
                pos.y0,
                cb_pos.width,
                pos.height
            ])

            # Get tight bounding boxes (includes labels/text)
            ax_bbox = ax.get_tightbbox(renderer)
            cb_bbox = cb_ax.get_tightbbox(renderer)

            # Merge axis + colorbar
            bbox = mpl.transforms.Bbox.union([ax_bbox, cb_bbox])

            # Convert to inches
            bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

            filename = f"{save_path}_panel_{i+1}.png"

            fig.savefig(filename, dpi=300, bbox_inches=bbox)

            # Reset colorbar position so the full figure is unaffected
            cb_ax.set_position(cb_pos)


    # Adjust spacing
    fig.subplots_adjust(wspace=0.1, bottom=0.18)

    # Save full figure
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    # plt.show()



            




