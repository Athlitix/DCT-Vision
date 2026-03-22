"""Native libjpeg-turbo bindings via cffi for direct DCT coefficient access.

Extracts and writes DCT coefficients without any pixel decode/encode.
Uses known struct offsets for libjpeg-turbo 2.x / libjpeg 8 ABI on x86_64 Linux.
"""

import os
import numpy as np
from cffi import FFI

from dct_vision.exceptions import InvalidImageError, LibjpegError

ffi = FFI()

ffi.cdef("""
    // Minimal declarations -- we access most fields via raw offsets

    // For jpeg_std_error
    typedef struct { char _opaque[168]; } jpeg_error_mgr;
    jpeg_error_mgr *jpeg_std_error(jpeg_error_mgr *err);

    // Opaque structs -- we allocate the right size and access fields by offset
    typedef struct { char _data[656]; } jpeg_decompress_struct;
    typedef struct { char _data[584]; } jpeg_compress_struct;

    // Virtual block array handle (opaque pointer)
    typedef struct jvirt_barray_control *jvirt_barray_ptr;

    // Block types
    typedef short JCOEF;
    typedef JCOEF JBLOCK[64];
    typedef JBLOCK *JBLOCKROW;
    typedef JBLOCKROW *JBLOCKARRAY;

    // API functions
    void jpeg_CreateDecompress(jpeg_decompress_struct *cinfo, int version, size_t sz);
    void jpeg_destroy_decompress(jpeg_decompress_struct *cinfo);
    void jpeg_stdio_src(jpeg_decompress_struct *cinfo, void *infile);
    int jpeg_read_header(jpeg_decompress_struct *cinfo, int require_image);
    jvirt_barray_ptr *jpeg_read_coefficients(jpeg_decompress_struct *cinfo);
    int jpeg_finish_decompress(jpeg_decompress_struct *cinfo);

    void jpeg_CreateCompress(jpeg_compress_struct *cinfo, int version, size_t sz);
    void jpeg_destroy_compress(jpeg_compress_struct *cinfo);
    void jpeg_stdio_dest(jpeg_compress_struct *cinfo, void *outfile);
    void jpeg_copy_critical_parameters(jpeg_decompress_struct *src, jpeg_compress_struct *dst);
    void jpeg_write_coefficients(jpeg_compress_struct *cinfo, jvirt_barray_ptr *coef_arrays);
    void jpeg_finish_compress(jpeg_compress_struct *cinfo);

    // libc
    void *fopen(const char *path, const char *mode);
    int fclose(void *fp);
""")

_lib = ffi.dlopen("jpeg")
_libc = ffi.dlopen(None)

_JPEG_LIB_VERSION = 80
DCTSIZE = 8
DCTSIZE2 = 64

# Known field offsets (libjpeg-turbo 2.x, x86_64 Linux)
# Determined via offsetof() -- see benchmarks or build scripts
_OFF_ERR = 0          # jpeg_decompress_struct.err
_OFF_MEM = 8          # jpeg_decompress_struct.mem
_OFF_WIDTH = 48       # jpeg_decompress_struct.image_width
_OFF_HEIGHT = 52      # jpeg_decompress_struct.image_height
_OFF_NUM_COMP = 56    # jpeg_decompress_struct.num_components
_OFF_COMP_INFO = 304  # jpeg_decompress_struct.comp_info (pointer)
_OFF_QUANT_PTRS = 200 # jpeg_decompress_struct.quant_tbl_ptrs[4]

# jpeg_component_info offsets
_CI_H_SAMP = 8
_CI_V_SAMP = 12
_CI_QUANT_TBL_NO = 16
_CI_WIDTH_BLOCKS = 28
_CI_HEIGHT_BLOCKS = 32
_CI_SIZEOF = 96

# jpeg_memory_mgr offset for access_virt_barray function pointer
_MEM_ACCESS_VIRT_BARRAY = 64


def _read_u32(ptr, offset):
    return ffi.cast("unsigned int *", ffi.cast("char *", ptr) + offset)[0]

def _read_i32(ptr, offset):
    return ffi.cast("int *", ffi.cast("char *", ptr) + offset)[0]

def _read_ptr(ptr, offset):
    return ffi.cast("void **", ffi.cast("char *", ptr) + offset)[0]


def _validate_jpeg(path: str) -> None:
    if not os.path.exists(path):
        raise InvalidImageError(f"File not found: {path}")
    with open(path, "rb") as f:
        magic = f.read(2)
    if len(magic) < 2 or magic != b"\xff\xd8":
        raise InvalidImageError(f"Not a JPEG file: {path}")


def _get_access_fn(cinfo):
    """Get the access_virt_barray function pointer from the memory manager."""
    mem_ptr = _read_ptr(cinfo, _OFF_MEM)
    fn_ptr = ffi.cast("void *", ffi.cast("char *", mem_ptr) + _MEM_ACCESS_VIRT_BARRAY)
    access_fn = ffi.cast(
        "JBLOCKARRAY (*)(void *, jvirt_barray_ptr, unsigned int, unsigned int, int)",
        ffi.cast("void **", fn_ptr)[0]
    )
    return access_fn


def read_dct_coefficients_native(path: str) -> dict:
    """Extract DCT coefficients directly from JPEG via libjpeg.

    No pixel decode. No DCT computation. Coefficients come directly
    from the JPEG file's compressed data.
    """
    _validate_jpeg(path)

    err = ffi.new("jpeg_error_mgr *")
    cinfo = ffi.new("jpeg_decompress_struct *")

    _lib.jpeg_std_error(err)
    # Set err pointer at offset 0
    ffi.cast("void **", cinfo)[0] = err
    _lib.jpeg_CreateDecompress(cinfo, _JPEG_LIB_VERSION, 656)

    fp = _libc.fopen(path.encode(), b"rb")
    if fp == ffi.NULL:
        _lib.jpeg_destroy_decompress(cinfo)
        raise InvalidImageError(f"Cannot open file: {path}")

    try:
        _lib.jpeg_stdio_src(cinfo, fp)
        rc = _lib.jpeg_read_header(cinfo, 1)
        if rc != 1:
            raise InvalidImageError(f"Invalid JPEG header: {path}")

        width = _read_u32(cinfo, _OFF_WIDTH)
        height = _read_u32(cinfo, _OFF_HEIGHT)
        num_components = _read_i32(cinfo, _OFF_NUM_COMP)

        coef_arrays = _lib.jpeg_read_coefficients(cinfo)
        access_fn = _get_access_fn(cinfo)

        # Read comp_info pointer
        comp_info_ptr = _read_ptr(cinfo, _OFF_COMP_INFO)

        coefficients = []
        comp_info_list = []

        for ci in range(num_components):
            ci_base = ffi.cast("char *", comp_info_ptr) + ci * _CI_SIZEOF

            h_samp = ffi.cast("int *", ci_base + _CI_H_SAMP)[0]
            v_samp = ffi.cast("int *", ci_base + _CI_V_SAMP)[0]
            quant_tbl_no = ffi.cast("int *", ci_base + _CI_QUANT_TBL_NO)[0]
            bw = ffi.cast("unsigned int *", ci_base + _CI_WIDTH_BLOCKS)[0]
            bh = ffi.cast("unsigned int *", ci_base + _CI_HEIGHT_BLOCKS)[0]

            comp_info_list.append({
                "h_samp_factor": h_samp,
                "v_samp_factor": v_samp,
                "quant_tbl_no": quant_tbl_no,
                "width_in_blocks": bw,
                "height_in_blocks": bh,
            })

            comp_coeffs = np.zeros((bh, bw, DCTSIZE, DCTSIZE), dtype=np.int16)
            row_bytes = bw * DCTSIZE2 * 2  # bw blocks * 64 coeffs * 2 bytes

            for row in range(bh):
                block_array = access_fn(
                    ffi.cast("void *", cinfo), coef_arrays[ci], row, 1, 0
                )
                # block_array[0] points to bw contiguous JBLOCKs
                row_data = ffi.buffer(block_array[0], row_bytes)
                comp_coeffs[row] = np.frombuffer(
                    row_data, dtype=np.int16
                ).reshape(bw, DCTSIZE, DCTSIZE)

            coefficients.append(comp_coeffs)

        # Extract quantization tables
        quant_tables = []
        for i in range(4):
            qtbl_ptr_addr = ffi.cast("char *", cinfo) + _OFF_QUANT_PTRS + i * 8
            qtbl_ptr = ffi.cast("void **", qtbl_ptr_addr)[0]
            if qtbl_ptr == ffi.NULL:
                continue
            # quantval is at offset 0, array of 64 unsigned short (or unsigned int)
            # In libjpeg-turbo 2.x, JQUANT_TBL.quantval is unsigned short[64]
            qtable = np.zeros((DCTSIZE, DCTSIZE), dtype=np.float32)
            qvals = ffi.cast("unsigned short *", qtbl_ptr)
            for k in range(DCTSIZE2):
                qtable[k // DCTSIZE, k % DCTSIZE] = qvals[k]
            quant_tables.append(qtable)

        _lib.jpeg_finish_decompress(cinfo)

    except (InvalidImageError, LibjpegError):
        raise
    except Exception as e:
        raise LibjpegError(f"libjpeg error reading {path}: {e}")
    finally:
        _libc.fclose(fp)
        _lib.jpeg_destroy_decompress(cinfo)

    return {
        "coefficients": coefficients,
        "quant_tables": quant_tables,
        "width": width,
        "height": height,
        "num_components": num_components,
        "comp_info": comp_info_list,
    }


def write_dct_coefficients_native(
    src_path: str,
    dst_path: str,
    coefficients: list[np.ndarray],
) -> None:
    """Write DCT coefficients to JPEG via libjpeg lossless transcode.

    Reads src_path for JPEG parameters, replaces coefficients, writes to dst_path.
    No pixel decode/encode.
    """
    _validate_jpeg(src_path)

    src_err = ffi.new("jpeg_error_mgr *")
    src_cinfo = ffi.new("jpeg_decompress_struct *")
    _lib.jpeg_std_error(src_err)
    ffi.cast("void **", src_cinfo)[0] = src_err
    _lib.jpeg_CreateDecompress(src_cinfo, _JPEG_LIB_VERSION, 656)

    src_fp = _libc.fopen(src_path.encode(), b"rb")
    if src_fp == ffi.NULL:
        _lib.jpeg_destroy_decompress(src_cinfo)
        raise InvalidImageError(f"Cannot open: {src_path}")

    try:
        _lib.jpeg_stdio_src(src_cinfo, src_fp)
        _lib.jpeg_read_header(src_cinfo, 1)
        src_coef_arrays = _lib.jpeg_read_coefficients(src_cinfo)

        num_components = _read_i32(src_cinfo, _OFF_NUM_COMP)
        access_fn = _get_access_fn(src_cinfo)

        # Replace coefficients -- copy entire rows at once
        for ci in range(min(num_components, len(coefficients))):
            bh, bw = coefficients[ci].shape[:2]
            row_bytes = bw * DCTSIZE2 * 2
            for row in range(bh):
                block_array = access_fn(
                    ffi.cast("void *", src_cinfo), src_coef_arrays[ci], row, 1, 1
                )
                # Write entire row of blocks in one memmove
                row_data = np.ascontiguousarray(
                    coefficients[ci][row].astype(np.int16)
                )
                ffi.memmove(block_array[0], row_data.tobytes(), row_bytes)

        # Write
        dst_err = ffi.new("jpeg_error_mgr *")
        dst_cinfo = ffi.new("jpeg_compress_struct *")
        _lib.jpeg_std_error(dst_err)
        ffi.cast("void **", dst_cinfo)[0] = dst_err
        _lib.jpeg_CreateCompress(dst_cinfo, _JPEG_LIB_VERSION, 584)

        dst_fp = _libc.fopen(dst_path.encode(), b"wb")
        if dst_fp == ffi.NULL:
            _lib.jpeg_destroy_compress(dst_cinfo)
            raise LibjpegError(f"Cannot write: {dst_path}")

        try:
            _lib.jpeg_stdio_dest(dst_cinfo, dst_fp)
            _lib.jpeg_copy_critical_parameters(src_cinfo, dst_cinfo)
            _lib.jpeg_write_coefficients(dst_cinfo, src_coef_arrays)
            _lib.jpeg_finish_compress(dst_cinfo)
        finally:
            _libc.fclose(dst_fp)
            _lib.jpeg_destroy_compress(dst_cinfo)

        _lib.jpeg_finish_decompress(src_cinfo)

    except (InvalidImageError, LibjpegError):
        raise
    except Exception as e:
        raise LibjpegError(f"libjpeg error: {e}")
    finally:
        _libc.fclose(src_fp)
        _lib.jpeg_destroy_decompress(src_cinfo)
