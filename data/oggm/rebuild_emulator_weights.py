#!/usr/bin/env python3
"""
Rebuild the Aletsch SIADecompNet pretrained weights.h5 so it loads cleanly
under the *current* IGM SR.py architecture (Keras 3).

Why this exists (supersedes fix_emulator_h5.py):
    The original weights.weights.h5 was saved by an older SR.py in which the
    context blocks and the slide/res heads lived inside an anonymous ``layers``
    container (``layers/conv2d_1..27``), while only context_in and def_head_*
    were top-level named layers. The current SR.py exposes ALL 28 conv layers
    as named attributes (context_in, context_block_0_conv1 … context_block_8_conv2,
    slide_head_conv1/2/out, def_head_conv1/2/out, res_head_conv1/2/out), so
    ``model.load_weights`` on the old file fails ("Layer 'X' expected 2
    variables, but received 0"). fix_emulator_h5.py only relocated 6 of the
    layers, so context_in/context_blocks still tripped the loader.

Robust approach: the old h5 conveniently stores each layer's TRUE name in the
``vars`` subgroup ``name`` attribute. We read (layer_name -> [kernel, bias])
for every layer, build a fresh model exactly as artifacts.rebuild_* does,
assign the weights by matching layer name, then RE-SAVE with the current model.
The re-saved file's object graph matches the current code by construction, so
IGM loads it without any group surgery.

Output: <this repo>/SIADecompNet_18x80_dil1-32/{export/weights.weights.h5, manifest.yaml}
"""

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

SRC_DIR = Path("/home/klleshi/Documents/igm-examples/aletsch/SIADecompNet_18x80_dil1-32")
SRC_H5 = SRC_DIR / "export" / "weights.weights.h5"
DST_DIR = Path("/home/klleshi/Documents/igm-smb-inference/SIADecompNet_18x80_dil1-32")
DST_H5 = DST_DIR / "export" / "weights.weights.h5"


def collect_source_weights(h5_path):
    """name -> [kernel, bias] for every layer, keyed by vars.attrs['name']."""
    name2w = {}
    with h5py.File(h5_path, "r") as f:
        def visit(g):
            for k in g:
                it = g[k]
                if isinstance(it, h5py.Group):
                    v = it.get("vars")
                    if isinstance(v, h5py.Group) and len(v.keys()) > 0:
                        nm = v.attrs.get("name")
                        if nm:
                            arrs = [v[str(i)][()] for i in range(len(v.keys()))]
                            name2w[str(nm)] = arrs
                    visit(it)
        visit(f)
    return name2w


def build_skeleton():
    """Build the current model exactly as artifacts.rebuild_emulator_from_manifest."""
    from igm.processes.iceflow.emulate.utils import artifacts as A
    manifest = A.load_supported_manifest(SRC_DIR / "manifest.yaml")
    dtype = tf.float32
    ctor = dict(manifest.architecture.params)
    model = A.Architectures[str(manifest.architecture.name)](**ctor)
    model.input_normalizer = A.build_fixed_input_normalizer_from_manifest(
        manifest, dtype, expected_nb_inputs=manifest.nb_inputs, name="input_norm"
    )
    model.build(tf.TensorShape([None, 4, 4, manifest.nb_inputs]))
    return model, manifest


def main():
    name2w = collect_source_weights(SRC_H5)
    print(f"Collected {len(name2w)} named layers from {SRC_H5.name}")

    model, manifest = build_skeleton()

    # Assign weights by matching layer name across all nested conv layers.
    assigned, missing = [], []
    for layer in model.submodules:
        w = layer.get_weights()
        if len(w) != 2:                       # only conv layers (kernel+bias)
            continue
        nm = layer.name
        if nm in name2w:
            src = name2w[nm]
            cur = layer.get_weights()
            if [tuple(a.shape) for a in src] != [tuple(a.shape) for a in cur]:
                raise ValueError(
                    f"shape mismatch for {nm}: src={[a.shape for a in src]} "
                    f"cur={[a.shape for a in cur]}")
            layer.set_weights([np.asarray(a) for a in src])
            assigned.append(nm)
        else:
            missing.append(nm)

    print(f"Assigned {len(assigned)} layers.")
    if missing:
        raise SystemExit(f"UNMATCHED current layers (no source weights): {missing}")
    # Sanity: every source layer should have been consumed.
    unused = sorted(set(name2w) - set(assigned))
    if unused:
        print(f"  note: source layers not present in current model (ignored): {unused}")

    DST_H5.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(DST_H5))
    shutil.copy2(SRC_DIR / "manifest.yaml", DST_DIR / "manifest.yaml")
    print(f"Saved current-format weights -> {DST_H5}")
    print(f"Copied manifest            -> {DST_DIR / 'manifest.yaml'}")

    # Round-trip verify: fresh skeleton must load the re-saved file cleanly.
    model2, _ = build_skeleton()
    model2.load_weights(str(DST_H5))
    dummy = tf.zeros((1, 4, 4, manifest.nb_inputs), dtype=tf.float32)
    y = model2(dummy, training=False)
    print(f"Round-trip OK: reload + forward pass, output shape={tuple(y.shape)}")


if __name__ == "__main__":
    main()
