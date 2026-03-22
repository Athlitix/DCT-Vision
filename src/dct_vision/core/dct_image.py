"""DCTImage — core data structure representing an image as DCT coefficients."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from dct_vision.utils.constants import BLOCK_SIZE
from dct_vision.exceptions import InvalidImageError


class DCTImage:
    """Core data structure representing an image as DCT coefficients.

    Attributes
    ----------
    y_coeffs : np.ndarray
        Luminance DCT coefficients, shape (bh, bw, 8, 8), dtype int16.
    cb_coeffs : np.ndarray or None
        Blue-difference chroma coefficients (None for grayscale).
    cr_coeffs : np.ndarray or None
        Red-difference chroma coefficients (None for grayscale).
    quant_tables : list[np.ndarray]
        JPEG quantization tables, each shape (8, 8).
    width : int
        Original image width in pixels.
    height : int
        Original image height in pixels.
    num_components : int
        Number of color components (1 for grayscale, 3 for color).
    comp_info : list[dict]
        Per-component info (sampling factors, quant table index).
    """

    def __init__(
        self,
        y_coeffs: np.ndarray,
        cb_coeffs: np.ndarray | None,
        cr_coeffs: np.ndarray | None,
        quant_tables: list[np.ndarray],
        width: int,
        height: int,
        comp_info: list[dict] | None = None,
        source_path: str | None = None,
    ):
        self.y_coeffs = y_coeffs
        self.cb_coeffs = cb_coeffs
        self.cr_coeffs = cr_coeffs
        self.quant_tables = quant_tables
        self.width = width
        self.height = height
        self.num_components = 1 if cb_coeffs is None else 3
        self.comp_info = comp_info
        self._source_path = source_path

    def derive(
        self,
        y_coeffs: np.ndarray,
        cb_coeffs: np.ndarray | None = ...,
        cr_coeffs: np.ndarray | None = ...,
        width: int | None = None,
        height: int | None = None,
    ) -> DCTImage:
        """Create a new DCTImage inheriting metadata from this one.

        Use this when creating modified versions of an image to preserve
        source_path, quant_tables, and comp_info automatically.
        Pass ... (Ellipsis) for cb/cr to copy from self.
        """
        return DCTImage(
            y_coeffs=y_coeffs,
            cb_coeffs=self.cb_coeffs.copy() if cb_coeffs is ... else cb_coeffs,
            cr_coeffs=self.cr_coeffs.copy() if cr_coeffs is ... else cr_coeffs,
            quant_tables=self.quant_tables,
            width=width if width is not None else self.width,
            height=height if height is not None else self.height,
            comp_info=self.comp_info,
            source_path=self._source_path,
        )

    @classmethod
    def from_file(cls, path: str) -> DCTImage:
        """Load a JPEG file and extract DCT coefficients.

        Parameters
        ----------
        path : str
            Path to a JPEG file.

        Returns
        -------
        DCTImage
            Image represented as DCT coefficients.
        """
        try:
            from dct_vision._libjpeg.native import read_dct_coefficients_native
            result = read_dct_coefficients_native(path)
        except (ImportError, OSError):
            from dct_vision._libjpeg.bindings import read_dct_coefficients
            result = read_dct_coefficients(path)
        coeffs = result["coefficients"]

        y_coeffs = coeffs[0]
        cb_coeffs = coeffs[1] if len(coeffs) > 1 else None
        cr_coeffs = coeffs[2] if len(coeffs) > 2 else None

        return cls(
            y_coeffs=y_coeffs,
            cb_coeffs=cb_coeffs,
            cr_coeffs=cr_coeffs,
            quant_tables=result["quant_tables"],
            width=result["width"],
            height=result["height"],
            comp_info=result.get("comp_info"),
            source_path=path,
        )

    @classmethod
    def from_array(cls, pixels: np.ndarray, quality: int = 85) -> DCTImage:
        """Create a DCTImage from a pixel array.

        Parameters
        ----------
        pixels : np.ndarray
            Pixel array, shape (H, W) for grayscale or (H, W, 3) for RGB.
            dtype uint8.
        quality : int
            JPEG quality factor (1-100) for quantization.

        Returns
        -------
        DCTImage
        """
        from dct_vision.math.colorspace import rgb_to_ycbcr
        from dct_vision.math.dct import blockwise_dct
        from dct_vision.math.quantization import scale_quant_table
        from dct_vision.core.block import pad_to_block_multiple
        from dct_vision.utils.constants import LUMINANCE_QUANT_TABLE, CHROMINANCE_QUANT_TABLE

        height, width = pixels.shape[:2]
        is_grayscale = pixels.ndim == 2

        luma_qtable = scale_quant_table(LUMINANCE_QUANT_TABLE, quality)
        chroma_qtable = scale_quant_table(CHROMINANCE_QUANT_TABLE, quality)

        if is_grayscale:
            channel = pad_to_block_multiple(pixels.astype(np.float32) - 128.0)
            raw_coeffs = blockwise_dct(channel)
            y_coeffs = np.round(raw_coeffs / luma_qtable).astype(np.int16)
            return cls(
                y_coeffs=y_coeffs,
                cb_coeffs=None,
                cr_coeffs=None,
                quant_tables=[luma_qtable],
                width=width,
                height=height,
                comp_info=[{"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0}],
            )

        # Color
        rgb = pixels.astype(np.float32)
        ycbcr = rgb_to_ycbcr(rgb)

        channels_coeffs = []
        qtables_list = [luma_qtable, chroma_qtable]

        for ch_idx in range(3):
            channel = ycbcr[:, :, ch_idx] - 128.0
            padded = pad_to_block_multiple(channel)
            raw_coeffs = blockwise_dct(padded)
            qtable = luma_qtable if ch_idx == 0 else chroma_qtable
            quantized = np.round(raw_coeffs / qtable).astype(np.int16)
            channels_coeffs.append(quantized)

        comp_info = [
            {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 0},
            {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 1},
            {"h_samp_factor": 1, "v_samp_factor": 1, "quant_tbl_no": 1},
        ]

        return cls(
            y_coeffs=channels_coeffs[0],
            cb_coeffs=channels_coeffs[1],
            cr_coeffs=channels_coeffs[2],
            quant_tables=qtables_list,
            width=width,
            height=height,
            comp_info=comp_info,
        )

    def to_pixels(self) -> np.ndarray:
        """Reconstruct pixel array from DCT coefficients.

        Returns
        -------
        np.ndarray
            Pixel array, shape (H, W) for grayscale or (H, W, 3) for RGB, dtype uint8.
        """
        from dct_vision.math.dct import blockwise_idct
        from dct_vision.math.colorspace import ycbcr_to_rgb

        if self.num_components == 1:
            # Grayscale
            qtable = self.quant_tables[0]
            dequantized = self.y_coeffs.astype(np.float32) * qtable
            channel = blockwise_idct(dequantized)
            channel = channel + 128.0
            return np.clip(channel[: self.height, : self.width], 0, 255).astype(np.uint8)

        # Color: dequantize and IDCT each channel
        channels = []
        all_coeffs = [self.y_coeffs, self.cb_coeffs, self.cr_coeffs]

        for ch_idx, coeffs in enumerate(all_coeffs):
            qtable_idx = self.comp_info[ch_idx]["quant_tbl_no"] if self.comp_info else (0 if ch_idx == 0 else 1)
            qtable = self.quant_tables[min(qtable_idx, len(self.quant_tables) - 1)]
            dequantized = coeffs.astype(np.float32) * qtable
            channel = blockwise_idct(dequantized)
            channel = channel + 128.0

            # Upsample chroma if needed
            if ch_idx > 0 and self.comp_info:
                h_samp = self.comp_info[0]["h_samp_factor"]
                v_samp = self.comp_info[0]["v_samp_factor"]
                ch_h = self.comp_info[ch_idx]["h_samp_factor"]
                ch_v = self.comp_info[ch_idx]["v_samp_factor"]
                if v_samp > ch_v:
                    channel = np.repeat(channel, v_samp // ch_v, axis=0)
                if h_samp > ch_h:
                    channel = np.repeat(channel, h_samp // ch_h, axis=1)

            channels.append(channel[: self.height, : self.width])

        min_h = min(ch.shape[0] for ch in channels)
        min_w = min(ch.shape[1] for ch in channels)
        ycbcr = np.stack([ch[:min_h, :min_w] for ch in channels], axis=-1).astype(np.float32)
        rgb = ycbcr_to_rgb(ycbcr)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def save(self, path: str, quality: int | None = None) -> None:
        """Save DCT coefficients as a JPEG file.

        Uses native libjpeg lossless transcode when possible (source JPEG
        exists and dimensions unchanged). Falls back to Pillow encode otherwise.

        Parameters
        ----------
        path : str
            Output file path.
        quality : int, optional
            If None, uses the original quantization tables.
        """
        coefficients = [self.y_coeffs]
        if self.cb_coeffs is not None:
            coefficients.append(self.cb_coeffs)
        if self.cr_coeffs is not None:
            coefficients.append(self.cr_coeffs)

        # Try native lossless write if we have a source JPEG
        if self._source_path and quality is None:
            try:
                from dct_vision._libjpeg.native import write_dct_coefficients_native
                write_dct_coefficients_native(
                    self._source_path, path, coefficients,
                )
                return
            except (ImportError, OSError):
                pass

        # Fallback to Pillow-based write
        from dct_vision._libjpeg.bindings import write_dct_coefficients
        write_dct_coefficients(
            path,
            coefficients,
            self.quant_tables,
            self.width,
            self.height,
            self.num_components,
            self.comp_info,
        )
