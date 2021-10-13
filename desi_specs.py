"""
desi_specs.py
Author: Yao-Yuan Mao

References:
- https://github.com/desihub/desitarget/blob/master/py/desitarget/sv3/data/sv3_targetmask.yaml
- https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/tiles/TILEID/NIGHT/coadd-SPECTRO-TILEID-NIGHT.html
"""

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack, unique
from easyquery import Query, QueryMaker
from SAGA.utils import join_str_arr

__all__ = ["load_fits", "get_all_tiles_and_nights", "find_redshifts_and_specs", "pack_for_marz", "is_bright_time", "is_bgs_target", "is_lowz_target"]
__author__ = "Yao-Yuan Mao"


BASE_DIR = "/global/cfs/cdirs/desi/spectro/redux/daily/tiles"


def load_fits(filename, hdu=1):
    """
    Usage:
    t = load_fits("/path/to/fits", hdu=1)
    t1, t2 = load_fits("/path/to/fits", hdu=[1, 2])
    """
    if isinstance(hdu, int):
        needed_hdu = [hdu]
        return_tuple = False
    else:
        needed_hdu = hdu
        return_tuple = True
        
    if not all((isinstance(h, int) for h in needed_hdu)):
        raise ValueError
    
    hdu_list = fits.open(filename, cache=False, memmap=True)
    
    try:
        t = [hdu_list[hdu].data if hdu_list[hdu].is_image else Table(hdu_list[hdu].data, masked=False) for hdu in needed_hdu]
    finally:
        try:
            for hdu in needed_hdu:
                del hdu_list[hdu].data
            hdu_list.close()
            del hdu_list
        except:  # pylint: disable=bare-except  # noqa: E722
            pass
    if return_tuple:
        return tuple(t)
    return t.pop()


def _get_redshifts(filepath, target_ids=None):
    z1, z2 = load_fits(filepath, [1, 2])
    if "SV3_DESI_TARGET" not in z2.colnames:
        return
    z2.sort(["TARGETID", "NUM_ITER"])
    z2 = unique(z2, "TARGETID", keep="last")
    q = Query() if target_ids is None else QueryMaker.isin("TARGETID", target_ids)
    z1 = q.filter(z1, ["TARGETID", "Z", "ZERR", "ZWARN", "CHI2", "SPECTYPE", "DELTACHI2"])
    z2 = q.filter(z2, ["TARGETID", "TARGET_RA", "TARGET_DEC", "FLUX_R", "FLUX_G", "SHAPE_R", "OBSCONDITIONS", "NIGHT", "TILEID", "SV3_DESI_TARGET", "SV3_BGS_TARGET", "SV3_SCND_TARGET"])
    if len(z1) and len(z2):
        z = join(z1, z2, "TARGETID")
        if len(z):
            return z


def _get_specs(filepath, sorted_target_ids):
    target_ids_here = load_fits(filepath)["TARGETID"]
    idx = np.searchsorted(sorted_target_ids, target_ids_here)
    idx[idx >= len(sorted_target_ids)] = -1
    matched = target_ids_here == sorted_target_ids[idx]
    if not matched.any():
        return 
    return (
        np.asarray(target_ids_here[matched]), 
        np.concatenate(load_fits(filepath, [2, 7, 12])),  # B_WAVELENGTH, R_WAVELENGTH, Z_WAVELENGTH
        np.hstack([d[matched] for d in load_fits(filepath, [3, 8, 13])]),  # B_FLUX, R_FLUX, Z_FLUX
        np.hstack([d[matched] for d in load_fits(filepath, [4, 9, 14])]),  # B_IVAR, R_IVAR, Z_IVAR
    )


def get_all_tiles_and_nights(nights_since=20210405):
    """
    Returns a table of TILEID and NIGHT for all nights since `nights_since`
    """
    out = {"TILEID":[], "NIGHT":[]}
    for tileid in os.listdir(BASE_DIR):
        try:
            tileid = int(tileid)
        except ValueError:
            continue
        dirpath = "{}/{}".format(BASE_DIR, tileid)
        try:
            nights = os.listdir(dirpath)
        except (OSError, IOError):
            continue
        for night in nights:
            try:
                night = int(night)
            except ValueError:
                continue
            if night >= nights_since:
                out["TILEID"].append(tileid)
                out["NIGHT"].append(night)
    return Table(out)


def _loop_over_files(tileid, night):
    dirpath = "{}/{}/{}".format(BASE_DIR, tileid, night)
    for filename in os.listdir(dirpath):
        if filename.endswith(".fits"):
            yield filename.partition("-")[0], os.path.join(dirpath, filename)

            
def _filename_to_path(filename):
    tileid, night = map(int, filename.partition('.')[0].split('-')[2:4])
    return "{}/{}/{}/{}".format(BASE_DIR, tileid, night, filename)
    

is_bright_time = Query("(OBSCONDITIONS >> 9) % 2 > 0")
is_bgs_target = Query("SV3_BGS_TARGET > 0")
is_lowz_target = Query("(SV3_SCND_TARGET >> 15) % 8 > 0")
is_lowz = Query("Z < 0.05")
is_galaxy = QueryMaker.equals("SPECTYPE", "GALAXY")

def find_redshifts_and_specs(t=None, retrieve_specs=False, skip_redshifts=False, selection=is_lowz_target, exclude_bgs=False, all_lowz=False, **kwargs):
    """
    Takes a table `t` with columns "TILEID" and "NIGHT", and all redshifts for LOWZ targets. 
    Set `exclude_bgs` to True to exclude targets that overlap with BGS. 
    
    Alternatively, the input table can have a "TARGETID" column,
    in which case the function will find corresponding redshifts.
     
    Set `retrieve_specs` to True to also obtain the spectra. 
    If this case, the returned variables are:
        redshifts, specs_flux, specs_ivar, specs_wl, specs_targetid
    
    Note that the function will not verify if all requested targets are found.
    It will also not verify if the redshifts table is consistent with specs.
    """
    if t is None:
        t = Table(kwargs)

    filenames_known = "FILENAME" in t.colnames
    targets_known = "TARGETID" in t.colnames

    group_keys = ["FILENAME"] if filenames_known else ["TILEID", "NIGHT"]
    assert all(c in t.colnames for c in group_keys)
    if targets_known:
        t.sort(group_keys + ["TARGETID"])

    redshifts = []
    specs = []
    
    if skip_redshifts:
        if not (filenames_known and targets_known):
            raise ValueError("Must have FILENAME and TARGETID in the input table to skip redshift collection.")
        if not retrieve_specs:
            raise ValueError("Nothing to do!!")
        redshifts = t
    else:
        q = Query(selection)
        if all_lowz:
            q = q | (is_lowz & is_galaxy) 
        if exclude_bgs:
            q = q & (~is_bgs_target)

        for t1 in t.group_by(group_keys).groups:
            if filenames_known:
                file_iter = [("zbest", _filename_to_path(t1["FILENAME"][0]))]
            else:
                file_iter = _loop_over_files(t1["TILEID"][0], t1["NIGHT"][0])
            for filetype, filepath in file_iter:
                if filetype == "zbest":
                    data_this = _get_redshifts(filepath, t1["TARGETID"] if targets_known else None)
                    if data_this is not None and not targets_known:
                        data_this = q.filter(data_this)
                    if data_this is not None and len(data_this):
                        data_this["FILENAME"] = os.path.basename(filepath)
                        redshifts.append(data_this)

        redshifts = vstack(redshifts)
        print("Found {} redshifts".format(len(redshifts)))

    redshifts.sort(["FILENAME", "TARGETID"])

    if not retrieve_specs:
        return redshifts

    for redshifts_this in redshifts.group_by(["FILENAME"]).groups:
        filepath = _filename_to_path(redshifts_this["FILENAME"][0].replace("zbest-", "coadd-"))
        data_this = _get_specs(filepath, redshifts_this["TARGETID"])
        if data_this is not None:
            specs.append(data_this)

    specs_id = np.concatenate([t[0] for t in specs])
    specs_flux = np.vstack([t[2] for t in specs])
    specs_ivar = np.vstack([t[3] for t in specs])
    sorter = specs_id.argsort()
    assert len(specs_id) == len(specs_flux)
    assert len(specs_id) == len(specs_ivar)
    
    specs_wl = specs[0][1]
    assert all((t[1] == specs_wl).all() for t in specs)
    
    print("Found {} specs".format(len(specs_id)))
    if len(redshifts) == len(specs_id) and not (redshifts["TARGETID"] == specs_id).all():
        print("WARNING: TARGETID in redshifts does not match those in specs")

    return redshifts, specs_flux, specs_ivar, specs_wl, specs_id


def pack_for_marz(output_path, redshifts, specs_flux, specs_ivar, specs_wl, *args):
    """
    Pack redshift table and specs into marz format. 
    Example usage:
    data = find_redshifts_and_specs(t, retrieve_specs=True)
    pack_for_marz("/path/to/output.fits", *data)
    """
    if len(redshifts) != len(specs_flux):
        raise ValueError
    
    with np.errstate(divide="ignore"):
        mag = 22.5 - 2.5*np.log10(redshifts["FLUX_R"])
        
    t = Table(
        {
            "TYPE": np.where(specs_flux.any(axis=1), "P", ""),
            "COMMENT": redshifts["TARGETID"].astype(str),
            "RA": np.deg2rad(redshifts["TARGET_RA"]),
            "DEC": np.deg2rad(redshifts["TARGET_DEC"]),
            "MAGNITUDE": mag,
        }
    )
    t["NAME"] = join_str_arr(
        "Z=", redshifts["Z"].astype(np.float32).astype(str), 
        ",ZW=", redshifts["ZWARN"].astype(str), 
        ",T=", (np.log2(redshifts["SV3_SCND_TARGET"])-14).astype(np.int16).astype(str),
    )

    # VACUME WAVELEGNTH TO AIR WAVELEGNTH
    dwave = specs_wl / (1.0 + 2.735182e-4 + 131.4182 / specs_wl**2 + 2.76249e8 / specs_wl**4)
    
    with np.errstate(divide="ignore"):
        specs_var = 1.0 / specs_ivar
    
    fits.HDUList([
        fits.PrimaryHDU(specs_flux, do_not_scale_image_data=True),
        fits.ImageHDU(specs_var, name="variance", do_not_scale_image_data=True),
        fits.ImageHDU(dwave, name="wavelength", do_not_scale_image_data=True),
        fits.BinTableHDU(t, name="fibres"),
    ]).writeto(output_path, overwrite=True)
