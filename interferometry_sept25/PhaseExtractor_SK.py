# PhaseExtractor_SK.py
# Per-pair phase extraction with normalization, per-pair low-frequency background subtraction,
# paginated saving (10 pairs per page) of:
#   - Raw (top) + background (bottom)
#   - Detrended (after subtraction)
# Plus final average plot & AvgPhase.txt (no .npy files).
#
# by Sebastian Kalos (base structure) + additions

import os
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import cv2
import scipy.ndimage as ndimage

import FourierRegionSelector_SK as FourierRegionSelector_SK


# --------------------------- Normalization helpers ---------------------------

def normalize_phase_pairs(
    diffcube: np.ndarray,
    flip_strategy: str = "median",   # "none" | "median"
    offset_strategy: str = "median"  # "none" | "median"
):
    """
    Normalize a stack of per-pair phase maps for visual comparison and averaging.

    - flip_strategy="median": multiply whole map by -1 if its median < 0
    - offset_strategy="median": subtract the per-map median (remove piston)
    """
    diff = diffcube.copy().astype(np.float32)
    N = diff.shape[0]

    flip_flags = np.zeros(N, dtype=bool)
    med_before = np.median(diff, axis=(1, 2))

    if flip_strategy == "median":
        flip_flags = med_before < 0
        diff[flip_flags] *= -1.0

    offsets = np.zeros(N, dtype=np.float32)
    if offset_strategy == "median":
        offsets = np.median(diff, axis=(1, 2)).astype(np.float32)
        diff = diff - offsets[:, None, None]

    return diff, flip_flags, med_before, offsets


def _shared_clim_from_two(
    stack_a: np.ndarray,
    stack_b: np.ndarray,
    mode: str = "percentile",
    p: float = 99.0,
    fixed: tuple[float, float] | None = None
):
    if mode == "fixed" and fixed is not None:
        return fixed
    if mode == "auto":
        vmin = float(min(stack_a.min(), stack_b.min()))
        vmax = float(max(stack_a.max(), stack_b.max()))
        return vmin, vmax
    # percentile mode (symmetric on |data|)
    concat_abs = np.abs(np.concatenate([stack_a.ravel(), stack_b.ravel()]))
    hi = float(np.percentile(concat_abs, p))
    return -hi, hi


def _shared_clim_one(stack: np.ndarray, mode="percentile", p=99.0, fixed=None):
    if mode == "fixed" and fixed is not None:
        return fixed
    if mode == "auto":
        return float(stack.min()), float(stack.max())
    hi = float(np.percentile(stack, p))
    return -hi, hi


# ---------------------- Low-frequency background removal ---------------------

def _poly_design_matrix(x, y, order: int):
    """Build 2D polynomial design matrix up to 'order' (1..3 typical)."""
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((x ** i) * (y ** j))
    return np.stack(cols, axis=1)


def estimate_lowfreq_background(
    img: np.ndarray,
    method: str = "gaussian",     # "gaussian" | "median" | "poly"
    *,
    bg_sigma: float = 50.0,       # pixels (for gaussian)
    bg_size: int = 101,           # odd window size (for median)
    poly_order: int = 2,          # for "poly" (1..3 typical)
    poly_downsample: int = 4      # speed up LS by sampling every Nth pixel
) -> np.ndarray:
    """
    Estimate smooth low-frequency background surface for one phase map.
    """
    img = img.astype(np.float32, copy=False)

    if method == "gaussian":
        return ndimage.gaussian_filter(img, sigma=bg_sigma, mode="nearest")

    if method == "median":
        k = int(bg_size) if int(bg_size) % 2 == 1 else int(bg_size) + 1
        return ndimage.median_filter(img, size=k, mode="nearest")

    if method == "poly":
        H, W = img.shape
        ys, xs = np.mgrid[0:H, 0:W]
        step = max(1, int(poly_downsample))
        x = xs[::step, ::step].astype(np.float32).ravel()
        y = ys[::step, ::step].astype(np.float32).ravel()
        z = img[::step, ::step].astype(np.float32).ravel()

        x_n = 2 * (x / (W - 1)) - 1.0
        y_n = 2 * (y / (H - 1)) - 1.0

        M = _poly_design_matrix(x_n, y_n, poly_order)
        coeffs, *_ = np.linalg.lstsq(M, z, rcond=None)

        Xn = 2 * (xs.astype(np.float32) / (W - 1)) - 1.0
        Yn = 2 * (ys.astype(np.float32) / (H - 1)) - 1.0
        Mf = _poly_design_matrix(Xn.ravel(), Yn.ravel(), poly_order)
        bg = (Mf @ coeffs).reshape(H, W).astype(np.float32)
        return bg

    raise ValueError("bg_method must be 'gaussian', 'median', or 'poly'")


def subtract_lowfreq_background_stack(
    stack: np.ndarray,
    method: str = "gaussian",
    *,
    bg_sigma: float = 50.0,
    bg_size: int = 101,
    poly_order: int = 2,
    poly_downsample: int = 4,
    re_center: bool = True
):
    """
    Apply low-frequency background estimation+subtraction to each slice.
    Returns (stack_bg, stack_detrended)
    """
    N = stack.shape[0]
    stack_bg = np.zeros_like(stack, dtype=np.float32)
    stack_out = np.zeros_like(stack, dtype=np.float32)
    for i in range(N):
        bg = estimate_lowfreq_background(
            stack[i], method=method,
            bg_sigma=bg_sigma, bg_size=bg_size,
            poly_order=poly_order, poly_downsample=poly_downsample
        )
        resid = stack[i] - bg
        if re_center:
            resid = resid - np.median(resid)
        stack_bg[i] = bg
        stack_out[i] = resid.astype(np.float32)
    return stack_bg, stack_out


# ------------------- Paginated saving (and showing) helpers ------------------

def _save_and_show(fig, out_path: str, show: bool = True, dpi: int = 150):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def save_raw_vs_background_paginated(
    raw_stack: np.ndarray,        # normalized per-pair maps (BEFORE BG subtraction)
    bg_stack: np.ndarray,         # estimated low-frequency backgrounds
    saveloc: str,
    *,
    pairs_per_fig: int = 10,      # 10 pairs per saved figure
    ncols: int = 5,               # 5 columns -> 2 pair-rows per page when 10 pairs
    cmap: str = "RdBu",
    clim_mode: str = "percentile",  # "percentile" | "auto" | "fixed"
    clim_value: float = 99.0,
    clim_fixed: tuple[float, float] | None = None,
    dpi: int = 150,
    show_pages: bool = True,
    folder_name: str = "pairs_raw_vs_bg",
    prefix: str = "pairs_raw_vs_bg_page"
):
    """
    Save+show figures, each with up to `pairs_per_fig` pairs laid out as:
      TOP:    raw (normalized) map
      BOTTOM: background for that pair
    Files: {saveloc}/{folder_name}/{prefix}{page:02d}.png
    """
    assert raw_stack.shape == bg_stack.shape, "Stacks must have identical shapes"
    out_dir = os.path.join(saveloc, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    N = raw_stack.shape[0]
    # Global color limits across ALL pages for fair comparison
    if clim_mode == "fixed" and clim_fixed is not None:
        vmin, vmax = clim_fixed
    else:
        vmin, vmax = _shared_clim_from_two(raw_stack, bg_stack, mode=clim_mode, p=clim_value)

    total_pages = int(np.ceil(N / pairs_per_fig))
    page_paths = []

    for page in range(total_pages):
        start = page * pairs_per_fig
        end   = min(start + pairs_per_fig, N)
        count = end - start

        ncols_used = min(ncols, count)
        nrows_pairs = int(np.ceil(count / ncols_used))
        nrows = 2 * nrows_pairs

        figsize = (min(3 * ncols_used, 30), min(6 * nrows_pairs, 36))
        fig, axes = plt.subplots(nrows, ncols_used, figsize=figsize, dpi=dpi, constrained_layout=True)
        axes = np.atleast_2d(axes)
        if axes.ndim == 1:
            axes = axes[:, None]

        im = None
        for k in range(count):
            idx = start + k
            rp = k // ncols_used
            c  = k % ncols_used
            ax_top = axes[2 * rp, c]
            ax_bot = axes[2 * rp + 1, c]

            im = ax_top.imshow(raw_stack[idx], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_top.set_xticks([]); ax_top.set_yticks([])
            ax_top.set_title(f"Pair {idx} — raw", fontsize=9)

            ax_bot.imshow(bg_stack[idx], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_bot.set_xticks([]); ax_bot.set_yticks([])
            ax_bot.set_title(f"Pair {idx} — background", fontsize=9)

        total_slots = nrows_pairs * ncols_used
        for k in range(count, total_slots):
            rp = k // ncols_used
            c  = k % ncols_used
            axes[2 * rp, c].axis("off")
            axes[2 * rp + 1, c].axis("off")

        if im is not None:
            cbar = fig.colorbar(im, ax=axes, shrink=0.9)
            cbar.set_label("Phase [rad]")

        fig.suptitle(f"Raw (top) & Background (bottom) — Pairs {start}–{end-1}", fontsize=12)

        out_path = os.path.join(out_dir, f"{prefix}{page+1:02d}.png")
        _save_and_show(fig, out_path, show=show_pages, dpi=dpi)
        page_paths.append(out_path)

    return page_paths


def save_detrended_paginated(
    detr_stack: np.ndarray,       # background-subtracted maps
    saveloc: str,
    *,
    pairs_per_fig: int = 10,
    ncols: int = 5,
    cmap: str = "RdBu",
    clim_mode: str = "percentile",
    clim_value: float = 99.0,
    clim_fixed: tuple[float, float] | None = None,
    dpi: int = 150,
    show_pages: bool = True,
    folder_name: str = "pairs_after_subtraction",
    prefix: str = "pairs_after_sub_page"
):
    """
    Save+show figures, each with up to `pairs_per_fig` detrended maps.
    Files: {saveloc}/{folder_name}/{prefix}{page:02d}.png
    """
    out_dir = os.path.join(saveloc, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    N = detr_stack.shape[0]
    # Global color limits across ALL pages for fair comparison
    if clim_mode == "fixed" and clim_fixed is not None:
        vmin, vmax = clim_fixed
    else:
        vmin, vmax = _shared_clim_one(detr_stack.ravel(), mode=clim_mode, p=clim_value)

    total_pages = int(np.ceil(N / pairs_per_fig))
    page_paths = []

    for page in range(total_pages):
        start = page * pairs_per_fig
        end   = min(start + pairs_per_fig, N)
        count = end - start

        ncols_used = min(ncols, count)
        nrows = int(np.ceil(count / ncols_used))

        figsize = (min(3 * ncols_used, 30), min(3 * nrows, 30))
        fig, axes = plt.subplots(nrows, ncols_used, figsize=figsize, dpi=dpi, constrained_layout=True)
        axes = np.atleast_2d(axes)
        if axes.ndim == 1:
            axes = axes[None, :]

        im = None
        k = 0
        for r in range(nrows):
            for c in range(ncols_used):
                ax = axes[r, c]
                if k < count:
                    idx = start + k
                    im = ax.imshow(detr_stack[idx], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f"Pair {idx}", fontsize=9)
                    k += 1
                else:
                    ax.axis("off")

        if im is not None:
            cbar = fig.colorbar(im, ax=axes, shrink=0.9)
            cbar.set_label("Phase [rad]")

        fig.suptitle(f"After background subtraction — Pairs {start}–{end-1}", fontsize=12)

        out_path = os.path.join(out_dir, f"{prefix}{page+1:02d}.png")
        _save_and_show(fig, out_path, show=show_pages, dpi=dpi)
        page_paths.append(out_path)

    return page_paths


# --------------------------------- Extractor --------------------------------

def PhaseExtractor(
    RawInterferograms,
    numShots,
    saveloc,
    boundary_points_sig,
    show_plots_flag,
    diag_angle_deg,
    diag_dist,
    fourier_window_size,
    *,
    # Pagination / layout
    pairs_per_fig: int = 10,          # 10 pairs per page
    grid_ncols: int = 5,              # 5 columns -> 2 pair-rows per raw+bg page
    cmap: str = "RdBu",
    clim_mode: str = "percentile",    # "percentile" | "auto" | "fixed"
    clim_value: float = 99.0,
    clim_fixed: tuple[float, float] | None = None,
    # Normalization
    flip_strategy: str = "median",    # "none" | "median"
    offset_strategy: str = "median",  # "none" | "median"
    # Background removal
    bg_method: str = "gaussian",      # "gaussian" | "median" | "poly"
    bg_sigma: float = 50.0,           # for gaussian
    bg_size: int = 101,               # for median (odd)
    poly_order: int = 2,              # for poly
    poly_downsample: int = 4,         # for poly
    # Show the saved pages while saving?
    show_pages: bool = True,
    # Save final outputs?
    save_avg_plot: bool = True        # save AvgPhase.png
):
    """
    Returns:
        AvgPhase (background-subtracted & normalized),
        DiffCube (raw per-pair),
        DiffCubeNorm (after flip/offset),
        DiffCubeDetr (after flip/offset + background subtraction),
        BgCube (estimated background surfaces per pair)

    Writes:
        - {saveloc}/pairs_raw_vs_bg/pairs_raw_vs_bg_pageXX.png (shown too)
        - {saveloc}/pairs_after_subtraction/pairs_after_sub_pageXX.png (shown too)
        - {saveloc}/AvgPhase.png  (shown if save_avg_plot=True)
        - {saveloc}/AvgPhase.txt  (final background-subtracted average, ASCII)
    """
    # Unpack:
    RawInterferogramssig = np.double(RawInterferograms[0])
    RawInterferogramsbg  = np.double(RawInterferograms[1])

    # ---------- FFT & boundary picking on first signal frame ----------
    FdifftSig0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(RawInterferogramssig[0, :, :])))
    FdiabsfftSig0 = np.absolute(FdifftSig0)
    fourier_image_sig = np.log(FdiabsfftSig0 + 1)

    os.makedirs(saveloc, exist_ok=True)
    img_scaled_sig = cv2.normalize(
        fourier_image_sig, dst=None, alpha=0, beta=255,
        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imwrite(os.path.join(saveloc, '2DFFT_sig.png'), img_scaled_sig)

    if boundary_points_sig is None:
        boundary_points_sig = FourierRegionSelector_SK.show_mouse_select(
            os.path.join(saveloc, '2DFFT_sig.png'), fourier_window_size
        )
        print('these are the fourier boundary points: ' + str(boundary_points_sig))

    # (row_top, row_bot, col_left, col_right)
    boundary_sig = (
        boundary_points_sig[0][1], boundary_points_sig[1][1],
        boundary_points_sig[0][0], boundary_points_sig[1][0]
    )

    # ---------- Hann window over all frames ----------
    (nframes, nrows, ncolumns) = RawInterferogramssig.shape
    WindowFcSig = np.zeros((nframes, nrows, ncolumns), dtype=np.float32)
    r0, r1, c0, c1 = boundary_sig
    hann_rows = np.hanning(r1 - r0)
    hann_cols = np.hanning(c1 - c0)
    hann_2D = np.outer(hann_rows, hann_cols).astype(np.float32)
    WindowFcSig[:, r0:r1, c0:c1] = hann_2D

    # ---------- FFT of all frames ----------
    FdifftSig = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(RawInterferogramssig)), axes=(-2, -1))
    FdifftBg  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(RawInterferogramsbg)),  axes=(-2, -1))

    # Apply window
    Hamfdifft = FdifftSig * WindowFcSig
    Hambgfft  = FdifftBg  * WindowFcSig  # same window

    # Center selected lobe
    col_shift = int(nrows / 2)    - int(boundary_sig[0] + (boundary_sig[1] - boundary_sig[0]) / 2)
    row_shift = int(ncolumns / 2) - int(boundary_sig[2] + (boundary_sig[3] - boundary_sig[2]) / 2)
    Hamfdifft = np.roll(Hamfdifft, row_shift, axis=-1)
    Hamfdifft = np.roll(Hamfdifft, col_shift, axis=-2)
    Hambgfft  = np.roll(Hambgfft,  row_shift, axis=-1)
    Hambgfft  = np.roll(Hambgfft,  col_shift, axis=-2)

    # ---------- Diagonal noise masking ----------
    if diag_angle_deg >= 0:
        quadrants = ['Q1', 'Q3']
    else:
        quadrants = ['Q2', 'Q4']
    print("Masking quadrants:", quadrants)

    diag_angle_rad = diag_angle_deg * np.pi / 180.0
    x_offset = int(diag_dist * np.cos(abs(diag_angle_rad)))
    y_offset = int(diag_dist * np.sin(abs(diag_angle_rad)))

    x1 = int(ncolumns / 2) + x_offset; y1 = int(nrows / 2) + y_offset
    x2 = int(ncolumns / 2) - x_offset; y2 = int(nrows / 2) + y_offset
    x3 = int(ncolumns / 2) - x_offset; y3 = int(nrows / 2) - y_offset
    x4 = int(ncolumns / 2) + x_offset; y4 = int(nrows / 2) - y_offset

    side_len = 100
    mask = np.ones((nframes, nrows, ncolumns), dtype=np.float32)
    if 'Q1' in quadrants: mask[:, y1:y1 + side_len, x1:x1 + side_len] = 0
    if 'Q2' in quadrants: mask[:, y2:y2 + side_len, x2 - side_len:x2] = 0
    if 'Q3' in quadrants: mask[:, y3 - side_len:y3, x3 - side_len:x3] = 0
    if 'Q4' in quadrants: mask[:, y4 - side_len:y4, x4:x4 + side_len] = 0

    if show_plots_flag:
        image = np.log(np.absolute(Hamfdifft[0, :, :]) + 1)
        fig, ax = plt.subplots()
        ax.imshow(image, origin='lower')
        plt.title("FFT after windowing (first frame)")
        plt.show()

    Hamfdifft *= mask
    Hambgfft  *= mask

    # ---------- IFFT & unwrap ----------
    Fdiifft = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Hamfdifft, axes=(-2, -1))), axes=(-2, -1))
    Bgifft  = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(Hambgfft,  axes=(-2, -1))), axes=(-2, -1))

    Fdiphase = np.single(np.angle(Fdiifft))
    Bgphase  = np.single(np.angle(Bgifft))

    UnwrappedFdiPhase = np.zeros((nframes, nrows, ncolumns), dtype=np.float32)
    UnwrappedBgPhase  = np.zeros((nframes, nrows, ncolumns), dtype=np.float32)
    for i in range(nframes):
        try:
            print(numShots[i])
        except Exception:
            print(i)
        uf = unwrap_phase(Fdiphase[i]); uf -= np.median(uf)
        ub = unwrap_phase(Bgphase[i]);  ub -= np.median(ub)
        UnwrappedFdiPhase[i] = uf
        UnwrappedBgPhase[i]  = ub

    # ---------- Per-pair phase maps ----------
    DiffCube = UnwrappedFdiPhase - UnwrappedBgPhase          # raw per pair
    DiffCubeNorm, _, _, _ = normalize_phase_pairs(           # normalized per pair
        DiffCube, flip_strategy=flip_strategy, offset_strategy=offset_strategy
    )

    # ---------- Background subtraction per pair ----------
    BgCube, DiffCubeDetr = subtract_lowfreq_background_stack(
        DiffCubeNorm,
        method=bg_method,
        bg_sigma=bg_sigma,
        bg_size=bg_size,
        poly_order=poly_order,
        poly_downsample=poly_downsample,
        re_center=True
    )

    # ---------- SAVE & SHOW: paginated raw(top)+background(bottom) ----------
    saved_rawbg = save_raw_vs_background_paginated(
        DiffCubeNorm, BgCube, saveloc,
        pairs_per_fig=pairs_per_fig,
        ncols=grid_ncols,
        cmap=cmap,
        clim_mode=clim_mode,
        clim_value=clim_value,
        clim_fixed=clim_fixed,
        dpi=150,
        show_pages=show_pages,
        folder_name="pairs_raw_vs_bg",
        prefix="pairs_raw_vs_bg_page"
    )
    print(f"Saved {len(saved_rawbg)} pages to: {os.path.join(saveloc, 'pairs_raw_vs_bg')}")

    # ---------- SAVE & SHOW: paginated detrended (after subtraction) ----------
    saved_detr = save_detrended_paginated(
        DiffCubeDetr, saveloc,
        pairs_per_fig=pairs_per_fig,
        ncols=grid_ncols,
        cmap=cmap,
        clim_mode=clim_mode,
        clim_value=clim_value,
        clim_fixed=clim_fixed,
        dpi=150,
        show_pages=show_pages,
        folder_name="pairs_after_subtraction",
        prefix="pairs_after_sub_page"
    )
    print(f"Saved {len(saved_detr)} pages to: {os.path.join(saveloc, 'pairs_after_subtraction')}")

    # ---------- Final average (after background subtraction) ----------
    AvgPhase = np.mean(DiffCubeDetr, axis=0)

    # Save final average plot (and show)
    if save_avg_plot:
        # Robust, symmetric scaling based on detrended stack
        if clim_mode == "fixed" and clim_fixed is not None:
            vmin, vmax = clim_fixed
        elif clim_mode == "auto":
            vmin, vmax = float(AvgPhase.min()), float(AvgPhase.max())
        else:
            vmin, vmax = _shared_clim_one(DiffCubeDetr.ravel(), mode=clim_mode, p=clim_value)

        fig = plt.figure(figsize=(6, 5), dpi=150)
        plt.imshow(AvgPhase, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        plt.title("Final average phase map (after background subtraction)")
        plt.colorbar(label="Phase [rad]")
        plt.tight_layout()
        out_avg_plot = os.path.join(saveloc, "AvgPhase.png")
        _save_and_show(fig, out_avg_plot, show=True, dpi=150)
        print(f"Saved final average plot to: {out_avg_plot}")

    # Save ONLY the final background-subtracted average as text
    out_avg_txt = os.path.join(saveloc, 'AvgPhase.txt')
    try:
        np.savetxt(out_avg_txt, AvgPhase)
        print(f"Saved AvgPhase.txt to: {out_avg_txt}")
    except Exception as e:
        print("Failed to save AvgPhase.txt:", e)
        raise

    return AvgPhase, DiffCube, DiffCubeNorm, DiffCubeDetr, BgCube
