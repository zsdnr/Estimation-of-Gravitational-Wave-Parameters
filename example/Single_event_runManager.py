
import jax
import jax.numpy as jnp

from src.jimgw.single_event.runManager import (SingleEventPERunManager,
                                           SingleEventRun)


jax.config.update("jax_enable_x64", True)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
mass_matrix = mass_matrix * 3e-3
local_sampler_arg = {"step_size": mass_matrix}
bounds = jnp.array(
    [
        [10.0, 40.0],
        [0.125, 1.0],
        [-1.0, 1.0],
        [-1.0, 1.0],
        [0.0, 2000.0],
        [-0.05, 0.05],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
        [0.0, jnp.pi],
        [0.0, 2 * jnp.pi],
        [-1.0, 1.0],
    ]
)


run = SingleEventRun(
    seed=0,
    path="test_data/GW150914/",
    detectors=["H1", "L1"],
    priors={
        "M_c": {"name": "Uniform", "xmin": 10.0, "xmax": 80.0},
        "q": {"name": "MassRatio"},
        "s1_z": {"name": "Uniform", "xmin": -1.0, "xmax": 1.0},
        "s2_z": {"name": "Uniform", "xmin": -1.0, "xmax": 1.0},
        "d_L": {"name": "Uniform", "xmin": 0.0, "xmax": 2000.0},
        "t_c": {"name": "Uniform", "xmin": -0.05, "xmax": 0.05},
        "phase_c": {"name": "Uniform", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "cos_iota": {"name": "CosIota"},
        "psi": {"name": "Uniform", "xmin": 0.0, "xmax": jnp.pi},
        "ra": {"name": "Uniform", "xmin": 0.0, "xmax": 2 * jnp.pi},
        "sin_dec": {"name": "SinDec"},
    },
    waveform_parameters={"name": "RippleIMRPhenomD", "f_ref": 20.0},
    jim_parameters={
        "n_loop_training": 10,
        "n_loop_production": 10,
        "n_local_steps": 150,
        "n_global_steps": 150,
        "n_chains": 500,
        "n_epochs": 50,
        "learning_rate": 0.001,
        "n_max_examples": 45000,
        "momentum": 0.9,
        "batch_size": 50000,
        "use_global": True,
        "keep_quantile": 0.0,
        "train_thinning": 1,
        "output_thinning": 10,
        "local_sampler_arg": local_sampler_arg,
    },
    likelihood_parameters={"name": "HeterodynedTransientLikelihoodFD", "bounds": bounds},
    injection=True,
    injection_parameters={
        "M_c": 28.6,
        "eta": 0.24,
        "s1_z": 0.05,
        "s2_z": 0.05,
        "d_L": 440.0,
        "t_c": 0.0,
        "phase_c": 0.0,
        "iota": 0.5,
        "psi": 0.7,
        "ra": 1.2,
        "dec": 0.3,
    },
    data_parameters={
        "trigger_time": 1126259462.4,
        "duration": 4,
        "post_trigger_duration": 2,
        "f_min": 20.0,
        "f_max": 1024.0,
        "tukey_alpha": 0.2,
        "f_sampling": 4096.0,
    },
)

run_manager = SingleEventPERunManager(run=run)
print("Single_event_runManager 进程已结束")

# 新加的
posterior_samples = run_manager.jim.sample(jax.random.PRNGKey(42))
run_manager.jim.print_summary()
run_manager.plot_injection_waveform("C:/Users/admin\Desktop/引力波/新任务/jim-main/src/plot")
run_manager.plot_data("C:/Users/admin\Desktop/引力波/新任务/jim-main/src/plot")

# print("posterior_samples:", posterior_samples)
# # 将列表中的 None 值替换为 0
# posterior_samples = [0 if x is None else x for x in posterior_samples]
# # 确保posterior_samples是numpy数组
# posterior_samples = jnp.array(posterior_samples)
# # 这两行代码的作用是过滤掉 posterior_samples 数组中的 NaN 和 inf 值，确保数组中只包含有效的数值，以便后续的处理和绘图。
# # posterior_samples = posterior_samples[~jnp.isnan(posterior_samples)]
# # posterior_samples = posterior_samples[~jnp.isinf(posterior_samples)]
# # 将 posterior_samples 数组中的 NaN 值替换为0
# posterior_samples = jnp.where(jnp.isnan(posterior_samples), 0, posterior_samples)
# # 将 posterior_samples 数组中的 inf 值替换为0
# # posterior_samples = jnp.where(jnp.isinf(posterior_samples), 0, posterior_samples)
# # 将数组中的所有正无穷值（inf）替换为0。jnp.isposinf 函数用于检测正无穷值。
# posterior_samples = jnp.where(jnp.isposinf(posterior_samples), 1, posterior_samples)
# # 将数组中的所有负无穷值（-inf）替换为0。jnp.isneginf 函数用于检测负无穷值。
# posterior_samples = jnp.where(jnp.isneginf(posterior_samples), -1, posterior_samples)
#
#
#
# # 生成图表
# import matplotlib.pyplot as plt
# plt.hist(posterior_samples, bins=50, density=True, alpha=0.7)
# plt.title("Posterior Distribution of Chirp Mass M_c")
# plt.xlabel("Chirp Mass [M_sun]")
# plt.ylabel("Probability Density")
# plt.show()