import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from jax import jacfwd, jacrev
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree
from tqdm import tqdm

from flowMC.proposal.base import ProposalBase
from typing import Callable

print("in BaseOn_Hessian_modified_MALA")


class MALA(ProposalBase):
    """
    Metropolis-adjusted Langevin algorithm sampler class
    building the mala_sampler method

    Args:
        logpdf: target logpdf function
        jit: whether to jit the sampler
        params: dictionary of parameters for the sampler
    """
    print("in BaseOn_Hessian_modified_MALA.MALA")

    step_size: Float

    def __init__(
            self,
            logpdf: Callable[[Float[Array, " n_dim"], PyTree], Float],
            jit: Bool,
            step_size: Float,
            use_autotune=False,
    ):
        super().__init__(logpdf, jit, step_size=step_size, use_autotune=use_autotune)
        self.step_size = step_size
        self.logpdf: Callable = logpdf
        self.use_autotune: Bool = use_autotune

    # def body(
    #         self,
    #         carry: tuple[Float[Array, " n_dim"], float, dict],
    #         this_key: PRNGKeyArray,
    # ) -> tuple[
    #     tuple[Float[Array, " n_dim"], float, dict],
    #     tuple[Float[Array, " n_dim"], Float[Array, "1"], Float[Array, " n_dim"], Float[Array, " n_dim n_dim"]],
    # ]:
    #     print("Compiling MALA body")
    #     this_position, dt, data = carry
    #     dt2 = dt * dt
    #     this_log_prob, this_d_log = jax.value_and_grad(self.logpdf)(this_position, data)
    #     this_hessian = jacfwd(jacrev(self.logpdf))(this_position, data)
    #     inv_hessian = jnp.linalg.inv(this_hessian)
    #
    #     proposal_mean = this_position + 0.5 * dt2 * jnp.dot(inv_hessian, this_d_log)
    #     proposal = proposal_mean + jnp.dot(jnp.linalg.cholesky(dt2 * inv_hessian),
    #                                        jax.random.normal(this_key, shape=this_position.shape))
    #
    #     return (proposal, dt, data), (proposal, this_log_prob, this_d_log, this_hessian)

    #
    def body(
            self,
            carry: tuple[Float[Array, " n_dim"], float, dict],
            this_key: PRNGKeyArray,
    ) -> tuple[
        tuple[Float[Array, " n_dim"], float, dict],
        tuple[Float[Array, " n_dim"], Float[Array, "1"], Float[Array, " n_dim"]],
    ]:
        print("Compiling MALA body")
        this_position, dt, data = carry
        # print(f"this_position shape 1: {this_position.shape}")
        dt2 = dt * dt
        # print(f"dt2 : {dt2}")
        this_log_prob, this_d_log = jax.value_and_grad(self.logpdf)(this_position, data)  # 原来的
        this_log_prob = self.logpdf(this_position, data)
        print(f"this_d_log shape: {this_d_log.shape}")
        print(f"this_log_prob shape: {this_log_prob.shape}")
        # print(f"this_position shape 2: {this_position.shape}")
        this_hessian = jacfwd(jacrev(self.logpdf))(this_position, data)
        # print(f"this_position shape 3: {this_position.shape}")
        inv_hessian = jnp.linalg.inv(this_hessian)
        # print(f"this_hessian shape: {this_hessian.shape}")
        # print(f"inv_hessian shape: {inv_hessian.shape}")
        # p = 0.5 * dt2 * jnp.dot(inv_hessian, this_d_log)  # 原来的
        p = 0.5 * jnp.dot(dt2, jnp.dot(inv_hessian, this_d_log))
        # print(f"p shape: {p.shape}")
        proposal_mean = this_position + p
        # print(f"proposal_mean shape: {proposal_mean.shape}")
        proposal = proposal_mean + jnp.dot(jnp.linalg.cholesky(dt2 * inv_hessian),
                                           jax.random.normal(this_key, shape=this_position.shape))
        # print(f"proposal shape 1: {proposal.shape}")
        # return (this_position, dt, data), (proposal, this_log_prob, this_d_log)  # 原来的
        return (this_position, dt, data), (proposal, this_log_prob, this_d_log, this_hessian, inv_hessian)

    # def kernel(
    #         self,
    #         rng_key: PRNGKeyArray,
    #         position: Float[Array, " n_dim"],
    #         log_prob: Float[Array, "1"],
    #         data: PyTree,
    # ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
    #     """
    #     Metropolis-adjusted Langevin algorithm kernel.
    #     This is a kernel that only evolve a single chain.
    #
    #     Args:
    #         rng_key (PRNGKeyArray): Jax PRNGKey
    #         position (Float[Array, " n_dim"]): current position of the chain
    #         log_prob (Float[Array, "1"]): current log-probability of the chain
    #         data (PyTree): data to be passed to the logpdf function
    #
    #     Returns:
    #         position (Float[Array, " n_dim"]): new position of the chain
    #         log_prob (Float[Array, "1"]): new log-probability of the chain
    #         do_accept (Int[Array, "1"]): whether the new position is accepted
    #     """
    #
    #     key1, key2 = jax.random.split(rng_key)
    #
    #     dt: Float = self.step_size
    #     dt2 = dt * dt
    #
    #     _, (proposal, logprob, d_logprob, hessian) = jax.lax.scan(
    #         self.body, (position, dt, data), jnp.array([key1, key1])
    #     )
    #
    #     inv_hessian = jnp.linalg.inv(hessian[0])
    #
    #     ratio = logprob[1] - logprob[0]
    #     ratio -= multivariate_normal.logpdf(
    #         proposal[0], position + 0.5 * dt2 * jnp.dot(inv_hessian, d_logprob[0]), dt2 * inv_hessian
    #     )
    #     ratio += multivariate_normal.logpdf(
    #         position, proposal[0] + 0.5 * dt2 * jnp.dot(jnp.linalg.inv(hessian[1]), d_logprob[1]),
    #                   dt2 * jnp.linalg.inv(hessian[1])
    #     )
    #
    #     log_uniform = jnp.log(jax.random.uniform(key2))
    #     do_accept: Bool[Array, " n_dim"] = log_uniform < ratio
    #
    #     position = jnp.where(do_accept, proposal[0], position)
    #     log_prob = jnp.where(do_accept, logprob[1], logprob[0])
    #
    #     return position, log_prob, do_accept

    #
    def kernel(
            self,
            rng_key: PRNGKeyArray,
            position: Float[Array, " n_dim"],
            log_prob: Float[Array, "1"],
            data: PyTree,
    ) -> tuple[Float[Array, " n_dim"], Float[Array, "1"], Int[Array, "1"]]:
        """
        Metropolis-adjusted Langevin algorithm kernel.
        This is a kernel that only evolve a single chain.

        Args:
            rng_key (PRNGKeyArray): Jax PRNGKey
            position (Float[Array, " n_dim"]): current position of the chain
            log_prob (Float[Array, "1"]): current log-probability of the chain
            data (PyTree): data to be passed to the logpdf function

        Returns:
            position (Float[Array, " n_dim"]): new position of the chain
            log_prob (Float[Array, "1"]): new log-probability of the chain
            do_accept (Int[Array, "1"]): whether the new position is accepted
        """

        key1, key2 = jax.random.split(rng_key)

        dt: Float = self.step_size
        dt2 = dt * dt
        # print("维度1：", position.shape, "position", position)
        # _, (proposal, logprob, d_logprob) = jax.lax.scan(
        #     self.body, (position, dt, data), jnp.array([key1, key1])
        # )  # 原来的
        _, (proposal, logprob, d_logprob, this_hessian, inv_hessian) = jax.lax.scan(
            self.body, (position, dt, data), jnp.array([key1, key1])
        )
        print(f"logprob shape: {logprob.shape}")
        print(f"logprob[1]]: {logprob[1]}")
        print(f"logprob[1]]: {logprob[0]}")
        # print(f"proposal shape 2: {proposal.shape}")
        # print("维度2：", position.shape, "position", position)
        # this_hessian = jacfwd(jacrev(self.logpdf))(position, data)  # 原来的
        # inv_hessian = jnp.linalg.inv(this_hessian)  # 原来的
        # print("维度3：", position.shape, "position", position)
        ratio = logprob[1] - logprob[0]
        ratio -= multivariate_normal.logpdf(
            proposal[0], position + 0.5 * jnp.dot(dt2, jnp.dot(inv_hessian, d_logprob[0])), jnp.dot(dt2, inv_hessian)
        )
        # print("维度4：", position.shape, "position", position)
        ratio += multivariate_normal.logpdf(
            position, proposal[0] + 0.5 * jnp.dot(dt2, jnp.dot(jnp.linalg.inv(this_hessian), d_logprob[1])),
            jnp.dot(dt2, jnp.linalg.inv(this_hessian))
        )
        # print("维度5：", position.shape, "position", position)
        log_uniform = jnp.log(jax.random.uniform(key2))
        # print("维度6：", position.shape, "position", position)
        do_accept: Bool[Array, " n_dim"] = log_uniform < ratio
        # print(f"proposal[0] shape: {proposal[0].shape}")
        # print(f"position shape: {position.shape}")
        print(f"do_accept shape: {do_accept.shape}")
        position = jnp.where(do_accept, proposal[0], position)
        # print("维度7：", position.shape, "position", position)
        log_prob = jnp.where(do_accept, logprob[1], logprob[0])
        print(f"log_prob shape: {log_prob.shape}")
        # print("维度8：", position.shape, "position", position)
        return position, log_prob, do_accept

    # ============================================================================
    # def update(
    #     self, i, state
    # ) -> tuple[
    #     PRNGKeyArray,
    #     Float[Array, "nstep  n_dim"],
    #     Float[Array, "nstep 1"],
    #     Int[Array, "n_step 1"],
    #     PyTree,
    # ]:
    #     """
    #     Update function for the MALA sampler
    #
    #     Args:
    #         i (int): current step
    #         state (tuple): state array storing the kernel information
    #
    #     Returns:
    #         state (tuple): updated state array
    #     """
    #     key, positions, log_p, acceptance, data = state
    #     _, key = jax.random.split(key)
    #     new_position, new_log_p, do_accept = self.kernel(
    #         key, positions[i - 1], log_p[i - 1], data
    #     )
    #     positions = positions.at[i].set(new_position)
    #     log_p = log_p.at[i].set(new_log_p)
    #     acceptance = acceptance.at[i].set(do_accept)
    #     return (key, positions, log_p, acceptance, data)
    def update(
            self, i, state
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "nstep  n_dim"],
        Float[Array, "nstep 1"],
        Int[Array, "n_step 1"],
        PyTree,
    ]:
        """
        Update function for the MALA sampler

        Args:
            i (int): current step
            state (tuple): state array storing the kernel information

        Returns:
            state (tuple): updated state array
        """
        key, positions, log_p, acceptance, data = state
        print(f"log_p shape 0: {log_p.shape}")
        _, key = jax.random.split(key)
        new_position, new_log_p, do_accept = self.kernel(
            key, positions[i - 1], log_p[i - 1], data
        )
        print(f"log_p shape 01: {log_p.shape}")
        print(f"new_log_p shape: {new_log_p.shape}")
        # print("维度：", positions.at[i].shape, new_position.shape, "new_position", new_position, "positions", positions)
        positions = positions.at[i].set(new_position)
        print(f"log_p shape 1: {log_p.shape}")
        log_p = log_p.at[i].set(new_log_p)
        print(f"log_p shape 2: {log_p.shape}")
        acceptance = acceptance.at[i].set(do_accept)
        return (key, positions, log_p, acceptance, data)

    def sample(
            self,
            rng_key: PRNGKeyArray,
            n_steps: int,
            initial_position: Float[Array, "n_chains  n_dim"],
            data: PyTree,
            verbose: bool = False,
    ) -> tuple[
        PRNGKeyArray,
        Float[Array, "n_chains n_steps  n_dim"],
        Float[Array, "n_chains n_steps 1"],
        Int[Array, "n_chains n_steps 1"],
    ]:
        logp = self.logpdf_vmap(initial_position, data)
        n_chains = rng_key.shape[0]
        acceptance = jnp.zeros((n_chains, n_steps))
        all_positions = (
                            jnp.zeros((n_chains, n_steps) + initial_position.shape[-1:])
                        ) + initial_position[:, None]
        all_logp = jnp.zeros((n_chains, n_steps)) + logp[:, None]
        state = (rng_key, all_positions, all_logp, acceptance, data)
        if verbose:
            iterator_loop = tqdm(
                range(1, n_steps),
                desc="Sampling Locally",
                miniters=int(n_steps / 10),
            )
        else:
            iterator_loop = range(1, n_steps)
        for i in iterator_loop:
            state = self.update_vmap(i, state)
        return state[:-1]

    def mala_sampler_autotune(
            self, rng_key, initial_position, log_prob, data, params, max_iter=30
    ):
        """
        Tune the step size of the MALA kernel using the acceptance rate.

        Args:
            mala_kernel_vmap (Callable): A MALA kernel
            rng_key: Jax PRNGKey
            initial_position (n_chains,  n_dim): initial position of the chains
            log_prob (n_chains, ): log-probability of the initial position
            params (dict): parameters of the MALA kernel
            max_iter (int): maximal number of iterations to tune the step size
        """

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # type: ignore

        counter = 0
        position, log_prob, do_accept = self.kernel_vmap(
            rng_key, initial_position, log_prob, data
        )
        acceptance_rate = jnp.mean(do_accept)
        while (acceptance_rate <= 0.3) or (acceptance_rate >= 0.5):
            if counter > max_iter:
                print(
                    "Maximal number of iterations reached.\
                    Existing tuning with current parameters."
                )
                break
            if acceptance_rate <= 0.3:
                self.step_size *= 0.8
            elif acceptance_rate >= 0.5:
                self.step_size *= 1.25
            counter += 1
            position, log_prob, do_accept = self.kernel_vmap(
                rng_key, initial_position, log_prob, data
            )
            acceptance_rate = jnp.mean(do_accept)
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)  # type: ignore
        return params
