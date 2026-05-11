#!/usr/bin/env python3
"""
Fetch OGGM gridded data for a glacier (default: Grosser Aletschgletscher,
RGI60-11.01450) into data/oggm/<RGI_ID>/.

Pulls the IGM-flavoured L3 preprocessed gdir from the Bremen cluster which
already contains: topo (≈ SRTM Feb 2000 for European Alps), consensus and
millan thicknesses, ITS_LIVE 1985-2018 composite velocities, Hugonnet 2000-2020
dhdt, glacier_mask. We additionally augment with GlaThiDa GPR thickness.

Usage:
    python fetch_oggm.py [RGI_ID]
"""

import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RGI_ID = "RGI60-11.01450"   # Grosser Aletschgletscher
WORKDIR_ROOT = "/tmp/oggm_igm_work"


def main(rgi_id):
    out_dir = os.path.join(HERE, rgi_id)
    print(f"== Fetching OGGM data for {rgi_id}")
    print(f"   → target: {out_dir}")

    import oggm.cfg as cfg_oggm
    from oggm import workflow, utils
    from oggm.shop import hugonnet_maps, its_live, glathida as oggm_glathida

    cfg_oggm.initialize_minimal()
    cfg_oggm.PARAMS["continue_on_error"] = True
    cfg_oggm.PARAMS["use_multiprocessing"] = False
    cfg_oggm.PATHS["working_dir"] = utils.gettempdir(
        dirname=f"OGGM-igm-{os.getpid()}", reset=True, home=False
    )
    print(f"   working dir: {cfg_oggm.PATHS['working_dir']}")

    # IGM's old igm_v2 URL for RGI6 is gone; use OGGM's standard L3 elev_bands
    # W5E5 prepro instead and add the IGM-needed extras (hugonnet, itslive,
    # glathida) via shop tasks below. Only border values 10/80/160 are
    # published for this prepro; 80 is the closest match to IGM's default 40.
    base_url = ("https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/"
                "L3-L5_files/2023.3/elev_bands/W5E5")
    gdirs = workflow.init_glacier_directories(
        [rgi_id],
        prepro_border=80,
        from_prepro_level=3,
        prepro_base_url=base_url,
    )
    if not gdirs:
        sys.exit(f"OGGM returned no gdir for {rgi_id}")
    gdir = gdirs[0]
    print(f"   gdir: {gdir.dir}")

    # Inspect what's already in gridded_data; add what's missing.
    import xarray as xr
    grid_nc = gdir.get_filepath("gridded_data")
    with xr.open_dataset(grid_nc) as ds:
        present = set(ds.data_vars)
    print(f"   variables already present: {sorted(present)}")

    if "hugonnet_dhdt" not in present:
        print("   → adding hugonnet_dhdt …")
        workflow.execute_entity_task(hugonnet_maps.hugonnet_to_gdir, gdirs)
    if "itslive_vx" not in present:
        print("   → adding itslive velocity …")
        workflow.execute_entity_task(its_live.velocity_to_gdir, gdirs)
    if "millan_vx" not in present:
        print("   → adding millan22 velocity (fallback) …")
        from oggm.shop import millan22
        try:
            workflow.execute_entity_task(millan22.velocity_to_gdir, gdirs)
            workflow.execute_entity_task(millan22.thickness_to_gdir, gdirs)
        except Exception as e:
            print(f"   millan22 unavailable: {e}")

    print("   → adding GlaThiDa point thickness …")
    try:
        workflow.execute_entity_task(oggm_glathida.glathida_to_gdir, gdirs)
    except Exception as e:
        print(f"   GlaThiDa task failed: {e}")

    # Copy the gdir's gridded folder out so we don't depend on tmpdir longevity.
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    src = os.path.dirname(gdir.get_filepath("gridded_data"))
    shutil.copytree(src, out_dir)
    print(f"   copied → {out_dir}")

    # Quick sanity report on the final gridded_data.
    final_nc = os.path.join(out_dir, "gridded_data.nc")
    with xr.open_dataset(final_nc) as ds:
        print(f"\n== Final gridded_data variables ({final_nc}):")
        for v in ds.data_vars:
            da = ds[v]
            try:
                vmin = float(da.min().values)
                vmax = float(da.max().values)
                vmean = float(da.mean().values)
                rng = f"min={vmin:.3f} max={vmax:.3f} mean={vmean:.3f}"
            except Exception:
                rng = "<non-numeric>"
            print(f"  {v}: shape={tuple(da.shape)} units={da.attrs.get('units','')} "
                  f"long_name=\"{da.attrs.get('long_name','')}\"  {rng}")
        print(f"\n  attrs: pyproj_srs={ds.attrs.get('pyproj_srs','')!r}")


if __name__ == "__main__":
    rgi_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RGI_ID
    main(rgi_id)
