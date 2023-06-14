import pytest
import numpy as np

from metadynminer.hills import Hills
from metadynminer.fes import Fes

def read_hills(name, periodic, cv_per):
    return Hills(name, periodic=periodic, cv_per=cv_per)

def plumed_hills(filename, cvs, resolution=256):
    plumed_data = np.loadtxt(filename)
    plumed_data = np.reshape(plumed_data[:, cvs], [resolution] * cvs)
    plumed_data = plumed_data - np.min(plumed_data)
    return plumed_data

@pytest.mark.parametrize(
    "name, periodic, cv_per, resolution", [
        ("acealanme1d", [True], [[-np.pi, np.pi]], 256),
        ("acealanme", [True,True], [[-np.pi, np.pi], [-np.pi, np.pi]], 256),
        ("acealanme3d", [True,True,True], [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], 64)
    ]
)
def test_makefes(shared_datadir, name, periodic, cv_per, resolution):
    hill_name = shared_datadir / "hills" / name
    hills = read_hills(
        name=hill_name, periodic=periodic, cv_per=cv_per
    )
    metadynminer = Fes(hills, resolution=resolution).fes.T

    plumed_name = shared_datadir / f"plumed/{name}.dat"
    plumed = plumed_hills(plumed_name, len(periodic), resolution=resolution)
    assert np.mean(metadynminer) == pytest.approx(np.mean(plumed), abs=1)
    assert np.allclose(metadynminer, plumed, atol=4)

@pytest.mark.parametrize(
    "name, periodic, cv_per, resolution", [
        ("acealanme1d", [True], [[-np.pi, np.pi]], 256),
        ("acealanme", [True,True], [[-np.pi, np.pi], [-np.pi, np.pi]], 256),
        # ("acealanme3d", [True,True,True], [[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]], 64)
    ]
)
def test_makefes2(shared_datadir, name, periodic, cv_per, resolution):
    hill_name = shared_datadir / "hills" / name
    hills = read_hills(
        name=hill_name, periodic=periodic, cv_per=cv_per
    )
    metadynminer = Fes(hills, resolution=resolution, original=True).fes.T
    plumed_name = shared_datadir / f"plumed/{name}.dat"
    plumed = plumed_hills(plumed_name, len(periodic), resolution=resolution)
    assert np.mean(metadynminer) == pytest.approx(np.mean(plumed), abs=1e-3)
    assert np.allclose(metadynminer, plumed, atol=1e-2)
