import astropy.units as u
import numpy as np
import pytest, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_table_interpolation():

    from ctao_cosmic_ray_spectra.spectral import TableInterpolationSpectrum

    # log log
    energy = [1, 10, 100] * u.TeV
    flux = [50, 5, 0.5] / (u.GeV * u.m**2 * u.sr * u.s)

    spectrum = TableInterpolationSpectrum(energy, flux)
    assert u.isclose(spectrum(5 * u.TeV), 10 / (u.GeV * u.m**2 * u.sr * u.s))


    # lin lin
    energy = [1, 2, 3] * u.TeV
    flux = [10, 8, 6] / (u.GeV * u.m**2 * u.sr * u.s)

    spectrum = TableInterpolationSpectrum(energy, flux, log_energy=False, log_flux=False)
    assert u.isclose(spectrum(1.5 * u.TeV), 9 / (u.GeV * u.m**2 * u.sr * u.s))



def test_powerlaw_integrate_cone_invalid():
    from ctao_cosmic_ray_spectra.spectral import PowerLaw, point_source_flux_unit

    point_source = PowerLaw(
        normalization=1 * point_source_flux_unit,
        index=-2,
        e_ref=500 * u.GeV,
    )
    with pytest.raises(ValueError):
        point_source.integrate_cone(0 * u.deg, 2 * u.deg)


@pytest.mark.parametrize(
    "outer,expected",
    zip((90 * u.deg, 180 * u.deg), (2 * np.pi * u.sr, 4 * np.pi * u.sr))
)
def test_powerlaw_integrate_cone(outer, expected):
    from ctao_cosmic_ray_spectra.spectral import PowerLaw, point_source_flux_unit, diffuse_flux_unit
    diffuse_flux = PowerLaw(
        normalization=1 * diffuse_flux_unit,
        index=-2,
        e_ref=500 * u.GeV,
    )


    integrated = diffuse_flux.integrate_cone(0 * u.rad, outer)
    assert integrated.normalization.unit.is_equivalent(point_source_flux_unit)
    assert u.isclose(integrated.normalization, diffuse_flux.normalization * expected)
    assert integrated.index == diffuse_flux.index
    assert integrated.e_ref == diffuse_flux.e_ref


def test_powerlaw():
    from ctao_cosmic_ray_spectra.spectral import PowerLaw

    # check we get a reasonable unit out of astropy independent of input unit
    unit = u.TeV**-1 * u.m**-2 * u.s**-1
    power_law = PowerLaw(1e-10 * unit, -2.65)
    assert power_law(1 * u.TeV).unit == unit
    assert power_law(1 * u.GeV).unit == unit


def test_powerlaw_from_simulations():
    from ctao_cosmic_ray_spectra.simulations import SimulatedEventsInfo
    from ctao_cosmic_ray_spectra.spectral import PowerLaw

    # calculate sensitivity between 1 and 2 degrees offset from fov center
    obstime = 50 * u.hour

    simulated_events = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=10 * u.GeV,
        energy_max=100 * u.TeV,
        max_impact=1 * u.km,
        spectral_index=-2,
        viewcone_min=0 * u.deg,
        viewcone_max=0 * u.deg,
    )

    powerlaw = PowerLaw.from_simulation(simulated_events, obstime=obstime)
    assert powerlaw.index == -2
    # regression test, maybe better come up with an easy to analytically verify parameter combination?
    assert u.isclose(powerlaw.normalization, 1.76856511e-08 / (u.TeV * u.m**2 * u.s))


    simulated_events = SimulatedEventsInfo(
        n_showers=int(1e6),
        energy_min=10 * u.GeV,
        energy_max=100 * u.TeV,
        max_impact=1 * u.km,
        spectral_index=-2,
        viewcone_min=5 * u.deg,
        viewcone_max=10 * u.deg,
    )

    powerlaw = PowerLaw.from_simulation(simulated_events, obstime=obstime)
    assert powerlaw.index == -2
    # regression test, maybe better come up with an easy to analytically verify parameter combination?
    assert u.isclose(powerlaw.normalization, 2.471917427911683e-07 / (u.TeV * u.m**2 * u.s * u.sr))