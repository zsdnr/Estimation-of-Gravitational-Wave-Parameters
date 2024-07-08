# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import numpy as np
import jax.numpy as jnp
import jax

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import requests

from astropy.time import Time

from scipy.signal.windows import tukey
from scipy.interpolate import interp1d


from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar

from jaxgw.PE.detector_preset import *
from jaxgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from jaxgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

# ==========================================
trigger_time = 1187008882.43
seglen = 128

# determine segment bounds, placing trigger 2s before the end
post_trigger_duration = 2
start = trigger_time - seglen + post_trigger_duration
end = trigger_time + post_trigger_duration
# ==========================================
ifos = ['H1', 'L1', 'V1']
data_td_dict = {i: TimeSeries.fetch_open_data(i, start, end, version=2)
                for i in ifos}
# ==========================================
tukey_alpha = 0.00625
data_fd_dict = {}
for ifo, d in data_td_dict.items():
    w = tukey(len(d), tukey_alpha)
    f = np.fft.rfftfreq(len(d), d=d.dt)
    data_fd_dict[ifo] = FrequencySeries(np.fft.rfft(d*w)/d.dt, frequencies=f)
# ==========================================
psd_url = "https://dcc.ligo.org/public/0150/P1800061/011/GW170817_PSDs.dat"
with requests.get(psd_url) as r:
    psd_data = np.genfromtxt(r.iter_lines())
# ==========================================
psd_dict = {}
for i, (ifo, d) in enumerate(data_fd_dict.items()):
    p = interp1d(psd_data[:,0], psd_data[:,i+1], bounds_error=False,
                 fill_value=np.inf)
    psd_dict[ifo] = FrequencySeries(p(d.frequencies), frequencies=d.frequencies)
# ==========================================
from src.jimgw.PE.detector_preset import *
from src.jimgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector
from src.jimgw.PE.detector_projection import make_detector_response

H1 = get_H1()
H1_response = make_detector_response(H1[0], H1[1])
L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])
V1 = get_V1()
V1_response = make_detector_response(V1[0], V1[1])

def LogLikelihood(theta):
    theta = theta.at[1].set(theta[1]/(1+theta[1])**2) # convert q to eta
    theta = theta.at[7].set(jnp.arccos(theta[7])) # convert cos iota to iota
    theta = theta.at[10].set(jnp.arcsin(theta[10])) # convert cos dec to dec
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp_test, hc_test = gen_IMRPhenomD_polar(H1_frequency, theta_waveform, f_ref)
    align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5]))
    h_test_H1 = H1_response(H1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_L1 = L1_response(L1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_V1 = V1_response(V1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    df = H1_frequency[1] - H1_frequency[0]
    match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*H1_data)/H1_psd*df).real
    match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*L1_data)/L1_psd*df).real
    match_filter_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*V1_data)/V1_psd*df).real
    optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
    optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real
    optimal_SNR_V1 = 4*jnp.sum((jnp.conj(h_test_V1)*h_test_V1)/V1_psd*df).

