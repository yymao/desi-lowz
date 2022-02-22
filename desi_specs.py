"""
desi_specs.py
Author: Yao-Yuan Mao

References:
- https://github.com/desihub/desitarget
  - https://github.com/desihub/desitarget/blob/master/py/desitarget/data/targetmask.yaml
  - https://github.com/desihub/desitarget/blob/master/py/desitarget/sv3/data/sv3_targetmask.yaml
- https://desidatamodel.readthedocs.io
  - https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/tiles/GROUPTYPE/TILEID/GROUPID/redrock-SPECTROGRAPH-TILEID-GROUPID.html
  - https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_REDUX/SPECPROD/tiles/GROUPTYPE/TILEID/GROUPID/coadd-SPECTROGRAPH-TILEID-GROUPID.html
"""

import os
import numpy as np
from astropy.io import fits
from astropy.table import Table, join, vstack, unique
from easyquery import Query, QueryMaker

__all__ = [
    "load_fits", "get_all_tiles_and_nights", "find_redshifts_and_specs", "pack_for_marz",
    "is_bgs_target_sv3", "is_bgs_target_main", "is_lowz_target_sv3", "is_lowz_target_main",
    "is_bright_time", "is_lowz_target", "is_bgs_target", "is_lowz", "is_galaxy",
]
__author__ = "Yao-Yuan Mao"
__version__ = "0.2.1"


BASE_DIR = "/global/cfs/cdirs/desi/spectro/redux/everest/tiles/cumulative"


def _fill_not_finite(arr, fill_value=99.0):
    return np.where(np.isfinite(arr), arr, fill_value)


def join_str_arr(*arrays):
    arrays_iter = iter(arrays)
    a = next(arrays_iter)
    for b in arrays_iter:
        a = np.char.add(a, b)
    return a


def mw_xtinct(ebv, band):
    """
    https://www.legacysurvey.org/dr9/catalogs/
    https://arxiv.org/pdf/1012.4804.pdf
    """
    coeffs = {"G": 3.214, "R": 2.165, "i": 1.592, "Z": 1.211, "Y": 1.064}
    Ab = coeffs[band] * ebv
    return 10.0**(-Ab / 2.5)


def load_fits(filename, hdu=1, ignore_missing=False, apply_slice=None):
    """
    Usage:
    t = load_fits("/path/to/fits", hdu=1)
    t1, t2 = load_fits("/path/to/fits", hdu=[1, 2])
    t1, t2 = load_fits("/path/to/fits", hdu=["B_FLUX", "R_FLUX"])
    """
    if isinstance(hdu, (int, str)):
        needed_hdu = [hdu]
        return_tuple = False
    else:
        needed_hdu = list(hdu)
        return_tuple = True

    try:
        hdu_list = fits.open(filename, cache=False, memmap=True)

        if any(isinstance(hdu, str) for hdu in needed_hdu):
            names = [hdu.header.get("EXTNAME", "") for hdu in hdu_list]

        output = []
        for hdu in needed_hdu:
            if isinstance(hdu, int):
                try:
                    hdu_this = hdu_list[hdu]
                except IndexError:
                    if not ignore_missing:
                        raise IndexError("{} does not have HDU {}".format(filename, hdu))
                    continue
            elif isinstance(hdu, str):
                try:
                    hdu_idx = names.index(hdu)
                except ValueError:
                    if not ignore_missing:
                        raise IndexError("{} does not have HDU {}".format(filename, hdu))
                    continue
                hdu_this = hdu_list[hdu_idx]
            else:
                raise ValueError("Invalid hdu value:{}. Must be int or str".format(hdu))

            data = np.array(hdu_this.data) if hdu_this.is_image else Table(hdu_this.data, masked=False)
            if apply_slice is not None:
                data = data[apply_slice]
            output.append(data)
            del data, hdu_this.data, hdu_this

    finally:
        try:
            hdu_list.close()
            del hdu_list
        except:  # pylint: disable=bare-except  # noqa: E722
            pass

    if return_tuple:
        return tuple(output)
    return output.pop()


def _get_redshifts(filepath, night=None, target_ids=None):
    z1, z2 = load_fits(filepath, [1, 2])
    if not any(col in z2.colnames for col in ["SV3_DESI_TARGET", "DESI_TARGET"]):
        return
    # z2.sort(["TARGETID", "NUM_ITER"])
    z2 = unique(z2, "TARGETID", keep="last")
    q = Query() if target_ids is None else QueryMaker.isin("TARGETID", target_ids)
    z1 = q.filter(z1, ["TARGETID", "Z", "ZERR", "ZWARN", "CHI2", "DELTACHI2", "SPECTYPE"])
    z2_cols = [
        "TARGETID", "TARGET_RA", "TARGET_DEC", "FLUX_R", "FLUX_G", "SHAPE_R",
        "FLUX_IVAR_R", "FLUX_IVAR_G", "EBV", "OBSCONDITIONS", "NIGHT", "TILEID", "FIBER",
        "SV3_SCND_TARGET", "SV3_BGS_TARGET", "SV3_DESI_TARGET", "SCND_TARGET", "BGS_TARGET"
    ]
    z2_cols_exist = [col for col in z2_cols if col in z2.colnames]
    z2_cols_not_exist = [col for col in z2_cols if col not in z2.colnames]
    z2 = q.filter(z2, z2_cols_exist)
    for col in z2_cols_not_exist:
        z2[col] = np.int32(-1)
    if night is not None:
        z2["NIGHT"] = night
    if len(z1) and len(z2):
        z = join(z1, z2, "TARGETID")
        if len(z):
            for BAND in ("G", "R"):
                z[f"SIGMA_{BAND}"] = z[f"FLUX_{BAND}"] * np.sqrt(np.abs(z[f"FLUX_IVAR_{BAND}"]))
                z[f"MW_TRANSMISSION_{BAND}"] = mw_xtinct(z["EBV"], BAND)
            const = 2.5 / np.log(10)
            for band in "gr":
                BAND = band.upper()
                with np.errstate(divide="ignore", invalid="ignore"):
                    z[f"{band}_mag"] = _fill_not_finite(
                        22.5 - const * np.log(z[f"FLUX_{BAND}"] / z[f"MW_TRANSMISSION_{BAND}"])
                    )
                    z[f"{band}_err"] = _fill_not_finite(const / np.abs(z[f"SIGMA_{BAND}"]))
            return z


def _get_specs(filepath, target_ids_needed=None):
    target_ids = np.asarray(load_fits(filepath)["TARGETID"])

    if target_ids_needed is None:
        idx = None
    else:
        sorter = target_ids.argsort()
        idx = np.searchsorted(target_ids, target_ids_needed, "left", sorter=sorter)
        idx[idx >= len(target_ids)] = -1
        idx = sorter[idx]
        idx = idx[target_ids[idx] == target_ids_needed]
        if not len(idx):
            return

    # We use the order of "RBZ" because in the overlapping regions we prefer R.
    return (
        (target_ids if idx is None else target_ids[idx]),
        np.concatenate(load_fits(filepath, ["{}_WAVELENGTH".format(c) for c in "RBZ"], ignore_missing=True)),
        np.hstack(load_fits(filepath, ["{}_FLUX".format(c) for c in "RBZ"], ignore_missing=True, apply_slice=idx)),
        np.hstack(load_fits(filepath, ["{}_IVAR".format(c) for c in "RBZ"], ignore_missing=True, apply_slice=idx)),
    )


def get_all_tiles_and_nights(nights_since=None, cumulative_nights=True, base_dir=BASE_DIR):
    """
    Returns a table of TILEID and NIGHT for all nights since `nights_since`
    """
    out = {"TILEID": [], "NIGHT": []}
    for tileid in os.listdir(base_dir):
        try:
            tileid = int(tileid)
        except ValueError:
            continue
        dirpath = "{}/{}".format(base_dir, tileid)
        try:
            nights = os.listdir(dirpath)
        except (OSError, IOError):
            continue
        nights = [int(night) for night in nights if night.isdigit()]
        if not nights:
            continue
        if cumulative_nights:
            nights = [max(nights)]
        for night in nights:
            if nights_since is None or night >= nights_since:
                out["TILEID"].append(tileid)
                out["NIGHT"].append(night)
    return Table(out)


def _loop_over_files(tileid, night, base_dir=BASE_DIR):
    dirpath = "{}/{}/{}".format(base_dir, tileid, night)
    for filename in os.listdir(dirpath):
        if filename.endswith(".fits"):
            yield filename.partition("-")[0], os.path.join(dirpath, filename)


def _filename_to_path(filename, base_dir=BASE_DIR):
    tileid, night = filename.partition(".")[0].split("-")[2:4]
    tileid = int(tileid)
    night = int(night[4:] if night.startswith("thru") else night)
    return "{}/{}/{}/{}".format(base_dir, tileid, night, filename)


is_bright_time = Query("(OBSCONDITIONS >> 9) % 2 > 0")
is_bgs_target_sv3 = Query("SV3_BGS_TARGET > 0")
is_lowz_target_sv3 = Query("(SV3_SCND_TARGET >> 15) % 8 > 0")
is_bgs_target_main = Query("BGS_TARGET > 0")
is_lowz_target_main = Query("(SCND_TARGET >> 15) % 8 > 0")

is_lowz_target = is_lowz_target_sv3 | is_lowz_target_main
is_bgs_target = is_bgs_target_sv3 | is_bgs_target_main

is_lowz = Query("Z < 0.05")
is_galaxy = QueryMaker.equals("SPECTYPE", "GALAXY")

WAVELENGTHS_START = 3600.0
WAVELENGTHS_DELTA = 0.8
WAVELENGTHS_LEN = 7781


def find_redshifts_and_specs(t=None, retrieve_specs=False, exclude_bgs=False, skip_redshifts=False, all_lowz=False, selection=is_lowz_target, zcat_prefix="redrock", **kwargs):
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

    if skip_redshifts:
        if not (filenames_known and targets_known):
            raise ValueError("Must have FILENAME and TARGETID in the input table to skip redshift collection.")
        if not retrieve_specs:
            raise ValueError("Nothing to do!!")
        redshifts = t
    else:
        q = Query(selection)
        if all_lowz:
            q = q | Query(is_lowz, is_galaxy)
        if exclude_bgs:
            q = Query(q, ~is_bgs_target)

        redshifts = []
        for t1 in t.group_by(group_keys).groups:
            if filenames_known:
                file_iter = [(zcat_prefix, _filename_to_path(t1["FILENAME"][0]))]
            else:
                file_iter = _loop_over_files(t1["TILEID"][0], t1["NIGHT"][0])
            for filetype, filepath in file_iter:
                if filetype == zcat_prefix:
                    data_this = _get_redshifts(filepath, t1["NIGHT"][0], t1["TARGETID"] if targets_known else None)
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

    specs = []
    for redshifts_this in redshifts.group_by(["FILENAME"]).groups:
        filepath = _filename_to_path(redshifts_this["FILENAME"][0].replace(zcat_prefix + "-", "coadd-"))
        data_this = _get_specs(filepath, redshifts_this["TARGETID"])
        if data_this is not None:
            wl_idx = np.round((data_this[1] - WAVELENGTHS_START) / WAVELENGTHS_DELTA).astype(np.int64),
            wl_idx, arr_idx = np.unique(wl_idx, return_index=True)
            specs.append((
                data_this[0],
                wl_idx,
                data_this[2][:, arr_idx],
                data_this[3][:, arr_idx],
            ))

    specs_id = np.concatenate([t[0] for t in specs])
    specs_flux = np.zeros((len(specs_id), WAVELENGTHS_LEN), dtype=np.float32)
    specs_ivar = np.zeros_like(specs_flux)
    i = 0
    for _, wl_idx, flux, ivar in specs:
        n = len(flux)
        specs_flux[i:i + n, wl_idx] = flux
        specs_ivar[i:i + n, wl_idx] = ivar
        i += n
    specs_wl = np.linspace(WAVELENGTHS_START, WAVELENGTHS_START + WAVELENGTHS_DELTA * (WAVELENGTHS_LEN - 1), WAVELENGTHS_LEN)

    print("Found {} specs".format(len(specs_id)))
    if len(redshifts) == len(specs_id) and not (redshifts["TARGETID"] == specs_id).all():
        print("WARNING: TARGETID in redshifts does not match those in specs")

    return redshifts, specs_flux, specs_ivar, specs_wl, specs_id


def pack_for_marz(output_path, redshifts, specs_flux, specs_ivar, *args):
    """
    Pack redshift table and specs into marz format.
    Example usage:
    data = find_redshifts_and_specs(t, retrieve_specs=True)
    pack_for_marz("/path/to/output.fits", *data)
    """
    if len(redshifts) != len(specs_flux):
        raise ValueError("redshift table and specs image have different lengths")

    if "r_mag" in redshifts.colnames:
        mag = redshifts["r_mag"]
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            mag = 22.5 - 2.5 * np.log10(redshifts["FLUX_R"])

    t = Table(
        {
            "TYPE": np.where(specs_flux.any(axis=1), "P", ""),
            "COMMENT": redshifts["TARGETID"].astype(str),
            "RA": np.deg2rad(redshifts["TARGET_RA"]),
            "DEC": np.deg2rad(redshifts["TARGET_DEC"]),
            "MAGNITUDE": mag,
        }
    )
    scnd_target = np.where(redshifts["SCND_TARGET"] > 0, redshifts["SCND_TARGET"], redshifts["SV3_SCND_TARGET"])
    t["NAME"] = join_str_arr(
        "Z=", redshifts["Z"].astype(np.float32).astype(str),
        ",ZW=", redshifts["ZWARN"].astype(str),
        ",T=", (np.log2(scnd_target) - 14).astype(np.int16).astype(str),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        specs_var = 1.0 / specs_ivar

    primary_hdu = fits.PrimaryHDU(specs_flux, do_not_scale_image_data=True)
    primary_hdu.header["CRVAL1"] = WAVELENGTHS_START
    primary_hdu.header["CRPIX1"] = 0
    primary_hdu.header["CDELT1"] = WAVELENGTHS_DELTA
    primary_hdu.header["VACUUM"] = True

    fits.HDUList([
        primary_hdu,
        fits.ImageHDU(specs_var, name="variance", do_not_scale_image_data=True),
        fits.BinTableHDU(t, name="fibres"),
    ]).writeto(output_path, overwrite=True)
