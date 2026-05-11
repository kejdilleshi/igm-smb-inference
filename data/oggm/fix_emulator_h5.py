#!/usr/bin/env python3
"""
Fix the Aletsch SIADecompNet pretrained weights.h5 for current Keras 3 /
SR.py architecture.

The original weights file was saved when slide_head_* and res_head_* layers
were either built dynamically or under a different naming scheme, so their
variables landed under anonymous paths like ``layers/conv2d_19/vars/0`` in
the h5 instead of named groups (``slide_head_conv1/vars/0`` etc.) that the
current SR.py expects.

This script copies the source h5 to the local SIADecompNet folder and adds
top-level named groups that mirror the anonymous layer data, leaving the
def_head_*, context_in, and input_normalizer entries untouched.

Mapping (verified by Conv2D shape against SR.py architecture inspection):

    layers/conv2d_19  →  slide_head_conv1   (3, 3,  95, 80)
    layers/conv2d_20  →  slide_head_conv2   (3, 3,  80, 80)
    layers/conv2d_21  →  slide_head_out     (1, 1,  80,  2)
    layers/conv2d_25  →  res_head_conv1     (3, 3, 110, 40)
    layers/conv2d_26  →  res_head_conv2     (3, 3,  40, 40)
    layers/conv2d_27  →  res_head_out       (1, 1,  40,  4)
"""

import os
import shutil
import h5py

SRC_H5 = "/home/klleshi/Documents/igm-examples/aletsch/SIADecompNet_18x80_dil1-32/export/weights.weights.h5"
DST_DIR = "/home/klleshi/Documents/igm-smb-inference/SIADecompNet_18x80_dil1-32"
DST_H5 = os.path.join(DST_DIR, "export", "weights.weights.h5")

MAPPING = {
    "slide_head_conv1": "layers/conv2d_19",
    "slide_head_conv2": "layers/conv2d_20",
    "slide_head_out":   "layers/conv2d_21",
    "res_head_conv1":   "layers/conv2d_25",
    "res_head_conv2":   "layers/conv2d_26",
    "res_head_out":     "layers/conv2d_27",
}

EXPECTED_SHAPES = {
    "slide_head_conv1": ((3, 3,  95, 80), (80,)),
    "slide_head_conv2": ((3, 3,  80, 80), (80,)),
    "slide_head_out":   ((1, 1,  80,  2), (2,)),
    "res_head_conv1":   ((3, 3, 110, 40), (40,)),
    "res_head_conv2":   ((3, 3,  40, 40), (40,)),
    "res_head_out":     ((1, 1,  40,  4), (4,)),
}


def main():
    os.makedirs(os.path.dirname(DST_H5), exist_ok=True)
    shutil.copy2(SRC_H5, DST_H5)
    print(f"Copied {SRC_H5}\n    → {DST_H5}")

    with h5py.File(DST_H5, "a") as f:
        for new_name, src_path in MAPPING.items():
            kernel_shape = tuple(f[f"{src_path}/vars/0"].shape)
            bias_shape   = tuple(f[f"{src_path}/vars/1"].shape)
            exp_k, exp_b = EXPECTED_SHAPES[new_name]
            assert kernel_shape == exp_k, (
                f"{new_name}: expected kernel {exp_k}, got {kernel_shape} "
                f"at {src_path}/vars/0 — mapping may be wrong; bailing")
            assert bias_shape == exp_b, (
                f"{new_name}: expected bias {exp_b}, got {bias_shape}")

            if new_name in f:
                del f[new_name]
            grp = f.create_group(f"{new_name}/vars")
            f.copy(f[f"{src_path}/vars/0"], grp, name="0")
            f.copy(f[f"{src_path}/vars/1"], grp, name="1")
            # Keras 3 stores the layer name as an attribute on the vars
            # subgroup; without it, model.load_weights treats the layer as
            # if it had no variables in the file.
            grp.attrs["name"] = new_name
            print(f"  + {new_name}: kernel {kernel_shape}, bias {bias_shape}")

    print("Done. Verifying final structure …")
    with h5py.File(DST_H5, "r") as f:
        for name in MAPPING:
            k = tuple(f[f"{name}/vars/0"].shape)
            b = tuple(f[f"{name}/vars/1"].shape)
            print(f"  {name}/vars/0={k}, /vars/1={b}")

    # Copy the manifest.yaml alongside.
    src_manifest = os.path.join(os.path.dirname(os.path.dirname(SRC_H5)),
                                "manifest.yaml")
    dst_manifest = os.path.join(DST_DIR, "manifest.yaml")
    shutil.copy2(src_manifest, dst_manifest)
    print(f"\nCopied manifest → {dst_manifest}")


if __name__ == "__main__":
    main()
