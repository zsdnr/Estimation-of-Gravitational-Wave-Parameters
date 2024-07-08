import time

import jax
import jax.numpy as jnp
from gwosc.datasets import event_gps

from src.jimgw.jim import Jim
from src.jimgw.prior import Uniform
from src.jimgw.single_event.detector import H1, L1, V1
from src.jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from src.jimgw.single_event.waveform import RippleIMRPhenomD

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

gps = event_gps("GW170817")
duration = 128
post_trigger_duration = 32
start_pad = duration - post_trigger_duration
end_pad = post_trigger_duration
fmin = 20.0
fmax = 1024.0

ifos = ["H1", "L1"]#, "V1"]

H1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=4*duration, tukey_alpha=0.05, gwpy_kwargs={"version": 2, "cache": False})
L1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=4*duration, tukey_alpha=0.05, gwpy_kwargs={"version": 2, "cache": False})
# V1.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.05)

prior = Uniform(
    xmin=[1.18, 0.125, -0.3, -0.3, 1., -0.1, 0.0, -1, 0.0, 0.0, -1.0],
    xmax=[1.21, 1.0, 0.3, 0.3, 75., 0.1, 2 * jnp.pi, 1.0, jnp.pi, 2 * jnp.pi, 1.0],
    naming=[
        "M_c",
        "q",
        "s1_z",
        "s2_z",
        "d_L",
        "t_c",
        "phase_c",
        "cos_iota",
        "psi",
        "ra",
        "sin_dec",
    ],
    transforms={
        "q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2),
        "cos_iota": (
            "iota",
            lambda params: jnp.arccos(
                jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        ),
        "sin_dec": (
            "dec",
            lambda params: jnp.arcsin(
                jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
            ),
        ),
    },  # sin and arcsin are periodize cos_iota and sin_dec
)

likelihood = HeterodynedTransientLikelihoodFD(
    [H1],
    prior=prior,
    bounds=[prior.xmin, prior.xmax],
    waveform=RippleIMRPhenomD(),
    trigger_time=gps,
    duration=duration,
    post_trigger_duration=post_trigger_duration,
    n_loops=1000
)

# mass_matrix = jnp.eye(11)
# mass_matrix = mass_matrix.at[1, 1].set(1e-3)
# mass_matrix = mass_matrix.at[5, 5].set(1e-3)
# local_sampler_arg = {"step_size": mass_matrix * 3e-3}

# jim = Jim(
#     likelihood,
#     prior,
#     n_loop_training=100,
#     n_loop_production=10,
#     n_local_steps=150,
#     n_global_steps=150,
#     n_chains=500,
#     n_epochs=50,
#     learning_rate=0.001,
#     max_samples=45000,
#     momentum=0.9,
#     batch_size=50000,
#     use_global=True,
#     keep_quantile=0.0,
#     train_thinning=1,
#     output_thinning=10,
#     local_sampler_arg=local_sampler_arg,
# )

# jim.sample(jax.random.PRNGKey(42))
