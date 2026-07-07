"""Distribution-of-relaxation-times (DRT) impedance fit.

Vendored from ``battery_forecast/analysis/drt.py``. Requires ``hybrid-drt``
(imported as ``hybdrt``) and ``tqdm`` — installed via the optional ``[drt]``
extra. The oracle imports :func:`get_drt_impedance` lazily inside a
``try/except`` so DRT peaks are simply omitted when the extra is absent.
"""
import warnings

import numpy as np
from hybdrt.models import DRT
from tqdm import tqdm


def get_drt_impedance(frequency_data, impedance_data, time_data, trim=False, trim_freq=1e3):
    """
    This function takes the impedance data from the eis measurements as and input and returns
    the determined DRT, the EIS data, and the EIS fit.
    : param tuple frequency_data : tuple of the frequencies corresponding to the EIS data.
    : param tuple impedance_data : tuple of the EIS data.
    : return : returns 3 dictionaries, 1) DRT results, 2) Imedance data, 3) Impedance fit based on DRT
    Format of the dictionaries columns
    DRT results
    tau     DRT

    Impedance data
    Zreal   Zimag

    Impedance fit
    Zreal   Zimag
    """

    # Create a DRT instance for fitting
    drt = DRT()
    Zdata = {}
    DRTdict = {}
    Zfit = {}
    Freq = {}
    for i, k in tqdm(enumerate(impedance_data), total=len(impedance_data), desc='Computing DRT ...'):
        freq = frequency_data[i]
        time = time_data[i]
        # The minus sign is because the data file contains -Zimag.
        z = impedance_data[i]

        drt.fit_dop = True
        if trim:
            z = z[freq > trim_freq]
            freq = freq[freq > trim_freq]
        Zdata[i] = z
        # Fit the KK-trimmed data
        # fz_clean is a tuple of (frequencies, impedances)
        # Perform a Kramers-Kronig test to identify the valid frequency range of the measurement
        #
        # Both kk_test (internally) and fit_eis below run their own iterative
        # hybdrt fit and each independently warn "Solution did not converge
        # within max_iter iterations" whenever that fit hits its cap — hybdrt's
        # own message notes this is usually harmless (the fit still
        # completes), so silence it here rather than globally to avoid masking
        # unrelated UserWarnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="hybdrt.*")
            outlier_index, freq_lim, fz_clean = drt.kk_test(freq, z, show_plot=False)
            drt.fit_eis(*fz_clean, max_iter=50, error_structure="uniform", scale_data=True, penalty_type="integral")

        # drt.plot_results()
        Zfit[i] = drt.predict_z(freq)

        tau = drt.get_tau_eval(ppd=50)
        gamma = drt.predict_drt(tau=tau)
        DRTdict[i] = np.array([tau, gamma]).T

        Freq[i] = freq
    return DRTdict, Zdata, Zfit, Freq
