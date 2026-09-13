
from pyresample import geometry, kd_tree
import torch
import math
import numpy as np 


def collocate_swaths_neighbhors(
        lon_source, lat_source, data_source,
        lon_target, lat_target,
        radius_of_influence,
        neighbors,
        power,
        fill_value=float('nan'),
    ):
    # Convert PyTorch tensors to NumPy
    lon_source_np = lon_source.squeeze().cpu().numpy()
    lat_source_np = lat_source.squeeze().cpu().numpy()
    data_source_np = data_source.cpu().numpy()
    lon_target_np = lon_target.squeeze().cpu().numpy()
    lat_target_np = lat_target.squeeze().cpu().numpy()

    # Define swaths
    source_swath = geometry.SwathDefinition(lons=lon_source_np, lats=lat_source_np)
    target_swath = geometry.SwathDefinition(lons=lon_target_np, lats=lat_target_np)

    # Get neighbour info (index_array and distance_array will have shape (n_valid_output, k))
    valid_input_index, valid_output_index, index_array, distance_array = kd_tree.get_neighbour_info(
        source_geo_def=source_swath,
        target_geo_def=target_swath,
        radius_of_influence=radius_of_influence,
        neighbours=neighbors,
        nprocs=1,
    )

    # Resample each channel using pyresample's built-in IDW
    collocated_channels = []
    for c in range(data_source_np.shape[0]):
        resampled = kd_tree.get_sample_from_neighbour_info(
            resample_type='custom',
            output_shape=target_swath.shape,
            data=data_source_np[c],
            valid_input_index=valid_input_index,
            valid_output_index=valid_output_index,
            index_array=index_array,
            distance_array=distance_array,
            weight_funcs=lambda d: 1.0 / (d ** power + 1e-10),
            fill_value=fill_value,
        )
        collocated_channels.append(resampled)

    data_collocated = np.stack(collocated_channels, axis=0)
    return torch.tensor(data_collocated)

def collocate_swaths_nearest(
        lon_source, lat_source, data_source,
        lon_target, lat_target,
        radius_of_influence,
        fill_value=float('nan'),
    ):

    # Convert PyTorch tensors to NumPy
    lon_source_np = lon_source.squeeze().cpu().numpy()
    lat_source_np = lat_source.squeeze().cpu().numpy()
    data_source_np = data_source.cpu().numpy()

    lon_target_np = lon_target.squeeze().cpu().numpy()
    lat_target_np = lat_target.squeeze().cpu().numpy()

    
    # Define swaths
    source_swath = geometry.SwathDefinition(
        lons=lon_source_np,
        lats=lat_source_np
    )

    target_swath = geometry.SwathDefinition(
        lons=lon_target_np,
        lats=lat_target_np
    )
    
    collocated_channels = []
    for c in range(data_source_np.shape[0]):
        data_c = kd_tree.resample_nearest(
            source_geo_def=source_swath,
            target_geo_def=target_swath,
            data=data_source_np[c],
            fill_value=fill_value,
            radius_of_influence=radius_of_influence
        )

        collocated_channels.append(data_c)

    data_collocated = np.stack(collocated_channels, axis=0)
    return torch.tensor(data_collocated)

def collocate_swaths(
        lon_source, lat_source, data_source,
        lon_target, lat_target,
        radius_of_influence,
        neighbors,
        power,
        fill_value=float('nan'),
    ):

    if neighbors == 1:
        return collocate_swaths_nearest(
            lon_source, lat_source, data_source,
            lon_target, lat_target,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
        )
    else:
        return collocate_swaths_neighbhors(
            lon_source, lat_source, data_source,
            lon_target, lat_target,
            radius_of_influence=radius_of_influence,
            neighbors=neighbors,
            power=power,
            fill_value=fill_value,
        )



def consistent_nan_assign(lon, lat, data_tb, fill_value=float('nan')):
    # Do a sum on all cat channels to find nans on all channels at the same pixel
    all_cat = torch.cat([lon, lat, data_tb], dim=0)
    all_bad = torch.sum(all_cat, dim=0, keepdims=True)
    nan_indices = torch.isnan(all_bad)

    # Assign nan on all channels at the same pixel value if there is a single nan in any of the channels
    lon[nan_indices] = fill_value
    lat[nan_indices] = fill_value
    for ch_idx in range(data_tb.shape[0]):
        data_tb[ch_idx][nan_indices.squeeze()] = fill_value

    return lon, lat, data_tb
