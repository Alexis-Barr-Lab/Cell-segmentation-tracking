import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import dask.array as da
import zarr
import torch
from tifffile import imread, imwrite
from tqdm import tqdm

from ultrack import track, to_tracks_layer, tracks_to_zarr
from ultrack.config import MainConfig
from ultrack.imgproc import normalize
from ultrack.imgproc.segmentation import reconstruction_by_dilation, Cellpose
from ultrack.utils.array import array_apply, create_zarr
from ultrack.utils.cuda import import_module, to_cpu, torch_default_device
from ultrack.utils import estimate_parameters_from_labels, labels_to_contours
from ultrack.imgproc import tracks_properties
from pyift.shortestpath import watershed_from_minima
from skimage.segmentation import relabel_sequential
from skimage.filters import threshold_otsu
import skimage.morphology as morph
from cellpose import models

try:
    import cupy as xp
except ImportError:
    import numpy as xp


def remove_background(image, sigma=15.0):
    image = xp.asarray(image)
    ndi = import_module("scipy", "ndimage")
    seeds = ndi.gaussian_filter(image, sigma=sigma)
    background = reconstruction_by_dilation(seeds, image, iterations=100)
    foreground = np.maximum(image, background) - background
    return to_cpu(foreground)


def watershed_segm(frame, aux_labels, min_area):
    import numpy as np
    import scipy.ndimage as ndi
    from skimage.filters import threshold_otsu
    from skimage.segmentation import relabel_sequential
    from pyift.shortestpath import watershed_from_minima
    import skimage.morphology as morph

    frame = np.asarray(frame)
    aux_labels = np.asarray(aux_labels)
    disk3 = ndi.generate_binary_structure(frame.ndim, 3)
    frame = frame.astype(np.float32)
    frame = ndi.gaussian_filter(frame, 3.0)
    det = frame > (threshold_otsu(frame) * 0.75)
    det = np.logical_or(det, aux_labels > 0)
    det = morph.remove_small_objects(det, min_area)
    det = ndi.binary_closing(det, structure=disk3)
    edt = ndi.distance_transform_edt(det)
    labels = relabel_sequential(watershed_from_minima(-edt, det, H_minima=2.0)[1])[0]
    return labels


def gpu_find_boundaries(label_frame, mode='outer'):
    boundaries = xp.zeros_like(label_frame, dtype=xp.bool_)
    padded = xp.pad(label_frame, pad_width=1, mode='edge')
    center = padded[1:-1, 1:-1]
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor = padded[1+di:1+di+label_frame.shape[0],
                              1+dj:1+dj+label_frame.shape[1]]
            boundaries |= (center != neighbor)
    return boundaries.astype(xp.float32)


def gpu_labels_to_contours(labels, sigma=None):
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    shape = labels[0].shape
    for lb in labels:
        if lb.shape != shape:
            raise ValueError(f"All labels must have the same shape. Found {shape} and {lb.shape}")
    foreground = xp.zeros(shape, dtype=xp.bool_)
    contours = xp.zeros(shape, dtype=xp.float32)
    for t in range(shape[0]):
        fg_frame = xp.zeros(shape[1:], dtype=xp.bool_)
        ct_frame = xp.zeros(shape[1:], dtype=xp.float32)
        for lb in labels:
            lb_frame = lb[t]
            fg_frame |= (lb_frame > 0)
            ct_frame += gpu_find_boundaries(lb_frame, mode='outer')
        ct_frame /= len(labels)
        if sigma is not None:
            try:
                import cupyx.scipy.ndimage as cndi
            except ImportError:
                import scipy.ndimage as cndi
            ct_frame = cndi.gaussian_filter(ct_frame, sigma)
            max_val = ct_frame.max()
            if max_val > 0:
                ct_frame = ct_frame / max_val
        foreground[t] = fg_frame
        contours[t] = ct_frame
    return foreground, contours


def process_movie(movie_path):
    movie_path = Path(movie_path)
    movie_id = movie_path.stem
    print(f"Processing movie: {movie_id}")

    imgs = imread(movie_path)
    print(f"Image size (T, C, Y, X): {imgs.shape}")

    if imgs.ndim != 4:
        raise ValueError("Expected movie with 4D shape (T, C, Y, X)")

    T, C, Y, X = imgs.shape
    chunks = (1, Y, X)

    scratch_folder = Path(f"/home/to/scratch/folder/{movie_id}") #create a scratch folder for each movie
    scratch_folder.mkdir(parents=True, exist_ok=True)

    # Background removal and normalization for each channel
    normalized_paths = []
    foregrounds = []
    for c in range(C):
        ch_img = imgs[:, c, :, :]
        fg_path = scratch_folder / f"foreground_ch{c}.zarr"
        fg = create_zarr(ch_img.shape, ch_img.dtype, str(fg_path), chunks=chunks, overwrite=True)
        array_apply(ch_img, out_array=fg, func=remove_background, sigma=15.0, axis=0)
        foregrounds.append(fg)

        norm_path = scratch_folder / f"normalized_ch{c}.zarr"
        norm = create_zarr(ch_img.shape, np.float16, str(norm_path), chunks=chunks, overwrite=True)
        array_apply(fg, out_array=norm, func=normalize, gamma=0.5, axis=0)
        normalized_paths.append(norm_path)

    # Segmentation on nuclear channel
    nuclear_norm = zarr.open(str(normalized_paths[0]), mode="r")
    use_gpu = torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else torch.backends.mps.is_available()
    print("Cellpose GPU available:", use_gpu)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model='cpsam')
    
    def apply_cellpose(img):
        masks, _, _ = model.eval(img)
        return masks

    cellpose_labels_path = scratch_folder / "cellpose_labels.zarr"
    cp_labels = create_zarr(nuclear_norm.shape, np.uint16, str(cellpose_labels_path), chunks=chunks, overwrite=True)
    array_apply(nuclear_norm, out_array=cp_labels, func=apply_cellpose, axis=0)

    ws_labels_path = scratch_folder / "ws_labels.zarr"
    ws_labels = create_zarr(nuclear_norm.shape, np.int32, str(ws_labels_path), chunks=chunks, overwrite=True)
    array_apply(nuclear_norm, cp_labels, out_array=ws_labels, func=watershed_segm, min_area=250, axis=0)

    cp_np = zarr.open(str(cellpose_labels_path), mode="r")[:]
    ws_np = zarr.open(str(ws_labels_path), mode="r")[:]
    foreground_gpu, contours_gpu = gpu_labels_to_contours([xp.asarray(cp_np), xp.asarray(ws_np)], sigma=5.0)
    foreground_cpu = xp.asnumpy(foreground_gpu)
    contours_cpu = xp.asnumpy(contours_gpu)

    config = MainConfig()
    config.segmentation_config.n_workers = 8
    config.segmentation_config.min_area = 50
    config.segmentation_config.max_area = 950
    config.segmentation_config.min_frontier = 0.01
    config.segmentation_config.threshold = 0.5
    config.linking_config.max_neighbors = 5
    config.linking_config.max_distance = 40
    config.linking_config.n_workers = 8
    config.linking_config.z_score_threshold = 3.0
    config.tracking_config.division_weight = -0.01
    config.tracking_config.disappear_weight = -2
    config.tracking_config.appear_weight = -0.1
    config.tracking_config.window_size = 15
    config.tracking_config.overlap_size = 3
    config.tracking_config.solution_gap = 0.0
    config.data_config.working_dir = scratch_folder
    config.data_config.in_memory_db_id = hash(movie_id) % 10000

    track(config, detection=foreground_cpu, edges=contours_cpu, images=[nuclear_norm], overwrite=True)
    tracks_df, lineage_graph = to_tracks_layer(config)
    labels_final = tracks_to_zarr(config, tracks_df)

    output_folder = Path("/path/to/save/results")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Extract per-channel properties and combine
    prop_dfs = []
    for i, fg in enumerate(foregrounds):
        df = tracks_properties(labels_final, image=fg)
        df["channel"] = i
        prop_dfs.append(df)


    df_combined = pd.concat(prop_dfs, ignore_index=True)
    df_combined["time (min)"] = df_combined["t"] * 10

    id_cols = ["track_id", "t", "time (min)"]
    value_cols = [col for col in df_combined.columns if col not in id_cols + ["channel"]]

    df_wide = df_combined.pivot_table(
        index=id_cols,
        columns="channel",
        values=value_cols
    )

    
    df_wide.columns = [f"{col}_ch{ch}" for col, ch in df_wide.columns]
    df_wide = df_wide.reset_index()

    
    df_wide.to_csv(output_folder / f"properties_all_channels_{movie_id}.csv", index=False)


    tracks_df.to_csv(output_folder / f"tracks_ultrack_{movie_id}.csv", index=False)
    imwrite(output_folder / f"segments_ultrack_{movie_id}.tif", labels_final)
    imwrite(output_folder / f"cellpose_segments_{movie_id}.tif", cp_np)

    print(f"Saved results for {movie_id} into {output_folder}")

    gc.collect()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Process_movie_multichannel.py <movie_path>")
        sys.exit(1)
    movie_path = sys.argv[1]
    process_movie(movie_path)
