"""
Feature extractor for the DreamerV3 architecture based on the JAX implementation.
This code is adapted from the JAX implementation of DreamerV3 and is designed to
be used with Stable Baselines3 as a custom feature extractor.
Original Code:
https://github.com/danijar/dreamerv3
"""

from typing import Dict, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import optax

# from torch import nn
import elements

import re
import numpy as np
import ninjax as nj
import chex

from .utils.dreamerv3 import (
    rssm,
    heads,
    utils,
    opt,
    nets as nn,
)  # Import the JAX RSSM modules

f32 = jnp.float32
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(
    lambda x: x.sample(jax.random.PRNGKey(0)), xs
)  # Using a fixed key for sampling in the extractor
prefix = lambda xs, p: {f"{p}/{k}": v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


enc_simple_config = {
    "depth": 64,
    "mults": [2, 3, 4, 4],
    "layers": 3,
    "units": 1024,
    "act": "silu",
    "norm": "rms",
    "winit": "trunc_normal_in",
    "symlog": True,
    "outer": False,
    "kernel": 5,
    "strided": False,
    "name": "enc",
}

dyn_rssm_config = {
    "deter": 8192,
    "hidden": 1024,
    "stoch": 32,
    "classes": 64,
    "act": "silu",
    "norm": "rms",
    "unimix": 0.01,
    "outscale": 1.0,
    "winit": "trunc_normal_in",
    "imglayers": 2,
    "obslayers": 1,
    "dynlayers": 1,
    "absolute": False,
    "blocks": 8,
    "free_nats": 1.0,
    "name": "dyn",
}

dec_simple_config = {
    "depth": 64,
    "mults": [2, 3, 4, 4],
    "layers": 3,
    "units": 1024,
    "act": "silu",
    "norm": "rms",
    "outscale": 1.0,
    "winit": "trunc_normal_in",
    "outer": False,
    "kernel": 5,
    "bspace": 8,
    "strided": False,
    "name": "dec",
}

rewhead = {
    "layers": 1,
    "units": 1024,
    "act": "silu",
    "norm": "rms",
    "output": "symexp_twohot",
    "outscale": 0.0,
    "winit": "trunc_normal_in",
    "bins": 255,
}
conhead = {
    "layers": 1,
    "units": 1024,
    "act": "silu",
    "norm": "rms",
    "output": "binary",
    "outscale": 1.0,
    "winit": "trunc_normal_in",
}

policy = {
    "layers": 3,
    "units": 1024,
    "act": "silu",
    "norm": "rms",
    "minstd": 0.1,
    "maxstd": 1.0,
    "outscale": 0.01,
    "unimix": 0.01,
    "winit": "trunc_normal_in",
}
value = {
    "layers": 3,
    "units": 1024,
    "act": "silu",
    "norm": "rms",
    "output": "symexp_twohot",
    "outscale": 0.0,
    "winit": "trunc_normal_in",
    "bins": 255,
}
policy_dist_disc = "categorical"
policy_dist_cont = "bounded_normal"
imag_last = 0
imag_length = 15
horizon = 333
contdisc = True
imag_loss = {"slowtar": False, "lam": 0.95, "actent": 3e-4, "slowreg": 1.0}
repl_loss = {"slowtar": False, "lam": 0.95, "slowreg": 1.0}
slowvalue = {"rate": 0.02, "every": 1}
retnorm = {
    "impl": "perc",
    "rate": 0.01,
    "limit": 1.0,
    "perclo": 5.0,
    "perchi": 95.0,
    "debias": False,
}
valnorm = {"impl": "none", "rate": 0.01, "limit": 1e-8}
advnorm = {"impl": "none", "rate": 0.01, "limit": 1e-8}
reward_grad = True
repval_loss = True
repval_grad = True
report = True
report_gradnorms = False

loss_scales = {
    "rec": 1.0,
    "rew": 1.0,
    "con": 1.0,
    "dyn": 1.0,
    "rep": 0.1,
    "policy": 1.0,
    "value": 1.0,
    "repval": 0.3,
}
opt_config = {
    "lr": 4e-5,
    "agc": 0.3,
    "eps": 1e-20,
    "beta1": 0.9,
    "beta2": 0.999,
    "momentum": True,
    "wd": 0.0,
    "schedule": "const",
    "warmup": 1000,
    "anneal": 0,
}
ac_grads = False

replay_context = 1


class RSSMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor based on the RSSM (Recurrent State-Space Model) architecture
    from the provided JAX code.

    :param observation_space: Observation space
    :param features_dim: Number of features extracted. This should match the
        dimension of the concatenated deterministic and stochastic states.
    :param config: A dictionary containing the configuration for the RSSM and
        its components (encoder, decoder, dynamics model). This should mirror
        the relevant parts of the original `config` used in the `Agent` class.
    """

    def __init__(self, observation_space, features_dim: int):
        action_space = gym.spaces.Box(
            -1, 1, shape=(3,), dtype=np.float32
        )  # TODO: Feed in real action space
        super().__init__(observation_space, features_dim)

        exclude = ("is_first", "is_last", "is_terminal", "reward")
        enc_space = {k: v for k, v in observation_space.items() if k not in exclude}
        dec_space = {k: v for k, v in observation_space.items() if k not in exclude}

        # Instantiate JAX modules (we'll use dummy initializations here)
        self.enc = rssm.Encoder(enc_space, **enc_simple_config)
        self.dyn = rssm.Encoder(None, **dyn_rssm_config)  # act_space is None here
        self.dec = rssm.Encoder(dec_space, **dec_simple_config)

        # # Placeholder for learned parameters
        # self.params = {
        #     'enc': self.enc.init(jax.random.PRNGKey(0), jnp.zeros((1, *enc_space[list(enc_space.keys())[0]].shape), enc_space[list(enc_space.keys())[0]].dtype))['params'],
        #     'dyn': self.dyn.init(jax.random.PRNGKey(0), jnp.zeros((1, 1)), jnp.zeros((1, self.dyn.min_action)), training=False)['params'], # Dummy action
        #     'dec': self.dec.init(jax.random.PRNGKey(0), jnp.zeros((1, self.config.dyn[self.config.dyn.typ].deter + self.config.dyn[self.config.dyn.typ].stoch)), training=False)['params'],
        # }

        self.feat2tensor = lambda x: jnp.concatenate(
            [
                nn.cast(x["deter"]),
                nn.cast(x["stoch"].reshape((*x["stoch"].shape[:-2], -1))),
            ],
            -1,
        )

        # Determine output feature dimension
        # dummy_enc_output = self.enc.apply({'params': self.params['enc']}, jnp.zeros((1, *enc_space[list(enc_space.keys())[0]].shape), enc_space[list(enc_space.keys())[0]].dtype))
        # dummy_dyn_output = self.dyn.apply({'params': self.params['dyn']}, dummy_enc_output, jnp.zeros((1, self.dyn.min_action)), training=False)
        # self._features_dim = dummy_dyn_output['deter'].shape[-1] + dummy_dyn_output['stoch'].reshape(dummy_dyn_output['stoch'].shape[:-2] + (-1,)).shape[-1]

        scalar = elements.Space(np.float32, ())
        binary = elements.Space(bool, (), 0, 2)
        self.rew = heads.MLPHead(scalar, **rewhead, name="rew")
        self.con = heads.MLPHead(binary, **conhead, name="con")

        d1, d2 = policy_dist_disc, policy_dist_cont
        outs = {k: d1 if v.discrete else d2 for k, v in action_space.items()}
        self.pol = heads.MLPHead(action_space, outs, **policy, name="pol")

        self.val = heads.MLPHead(scalar, **value, name="val")
        self.slowval = utils.SlowModel(
            heads.MLPHead(scalar, **value, name="slowval"),
            source=self.val,
            **slowvalue,
        )

        self.retnorm = utils.Normalize(**retnorm, name="retnorm")
        self.valnorm = utils.Normalize(**valnorm, name="valnorm")
        self.advnorm = utils.Normalize(**advnorm, name="advnorm")

        self.modules = [
            self.dyn,
            self.enc,
            self.dec,
            self.rew,
            self.con,
            self.pol,
            self.val,
        ]
        self.opt = opt.Optimizer(
            self.modules, self._make_opt(**opt_config), summary_depth=1, name="opt"
        )

        scales = self.config.loss_scales.copy()
        rec = scales.pop("rec")
        scales.update({k: rec for k in dec_space})
        self.scales = scales

    @property
    def policy_keys(self):
        return "^(enc|dyn|dec|pol)/"

    @property
    def ext_space(self):
        spaces = {}
        spaces["consec"] = elements.Space(np.int32)
        spaces["stepid"] = elements.Space(np.uint8, 20)
        if replay_context:
            spaces.update(
                elements.tree.flatdict(
                    dict(
                        enc=self.enc.entry_space,
                        dyn=self.dyn.entry_space,
                        dec=self.dec.entry_space,
                    )
                )
            )
        return spaces

    def init_policy(self, batch_size):
        zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
        return (
            self.enc.initial(batch_size),
            self.dyn.initial(batch_size),
            self.dec.initial(batch_size),
            jax.tree.map(zeros, self.act_space),
        )

    def init_train(self, batch_size):
        return self.init_policy(batch_size)

    def init_report(self, batch_size):
        return self.init_policy(batch_size)

    def policy(self, carry, obs, mode="train"):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        kw = dict(training=False, single=True)
        reset = obs["is_first"]
        enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
        dyn_carry, dyn_entry, feat = self.dyn.observe(
            dyn_carry, tokens, prevact, reset, **kw
        )
        dec_entry = {}
        if dec_carry:
            dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
        policy = self.pol(self.feat2tensor(feat), bdims=1)
        act = sample(policy)
        out = {}
        out["finite"] = elements.tree.flatdict(
            jax.tree.map(
                lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
                dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act),
            )
        )
        carry = (enc_carry, dyn_carry, dec_carry, act)
        if replay_context:
            out.update(
                elements.tree.flatdict(
                    dict(enc=enc_entry, dyn=dyn_entry, dec=dec_entry)
                )
            )
        return carry, act, out
    

    def train(self, carry, data):
        carry, obs, prevact, stepid = self._apply_replay_context(carry, data)
        metrics, (carry, entries, outs, mets) = self.opt(
            self.loss, carry, obs, prevact, training=True, has_aux=True)
        metrics.update(mets)
        self.slowval.update()
        outs = {}
        if self.config.replay_context:
            updates = elements.tree.flatdict(dict(stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2]))
            B, T = obs['is_first'].shape
            assert all(x.shape[:2] == (B, T) for x in updates.values()), ((B, T), {k: v.shape for k, v in updates.items()})
            outs['replay'] = updates
        carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, outs, metrics


    def loss(self, carry, obs, prevact, training):
        enc_carry, dyn_carry, dec_carry = carry
        reset = obs['is_first']
        B, T = reset.shape
        losses = {}
        metrics = {}

        # World model
        enc_carry, enc_entries, tokens = self.enc(
            enc_carry, obs, reset, training)
        dyn_carry, dyn_entries, los, repfeat, mets = self.dyn.loss(
            dyn_carry, tokens, prevact, reset, training)
        losses.update(los)
        metrics.update(mets)
        dec_carry, dec_entries, recons = self.dec(
            dec_carry, repfeat, reset, training)
        inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
        losses['rew'] = self.rew(inp, 2).loss(obs['reward'])
        con = f32(~obs['is_terminal'])
        if self.config.contdisc:
            con *= 1 - 1 / self.config.horizon
        losses['con'] = self.con(self.feat2tensor(repfeat), 2).loss(con)
        for key, recon in recons.items():
            space, value = self.obs_space[key], obs[key]
            assert value.dtype == space.dtype, (key, space, value.dtype)
            target = f32(value) / 255 if isimage(space) else value
            losses[key] = recon.loss(sg(target))

        B, T = reset.shape
        shapes = {k: v.shape for k, v in losses.items()}
        assert all(x == (B, T) for x in shapes.values()), ((B, T), shapes)

        # Imagination
        K = min(self.config.imag_last or T, T)
        H = self.config.imag_length
        starts = self.dyn.starts(dyn_entries, dyn_carry, K)
        policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
        _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training)
        first = jax.tree.map(
            lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
        imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
        lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
        lastact = jax.tree.map(lambda x: x[:, None], lastact)
        imgact = concat([imgprevact, lastact], 1)
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
        assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
        inp = self.feat2tensor(imgfeat)
        los, imgloss_out, mets = imag_loss(
            imgact,
            self.rew(inp, 2).pred(),
            self.con(inp, 2).prob(1),
            self.pol(inp, 2),
            self.val(inp, 2),
            self.slowval(inp, 2),
            self.retnorm, self.valnorm, self.advnorm,
            update=training,
            contdisc=self.config.contdisc,
            horizon=self.config.horizon,
            **self.config.imag_loss)
        losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
        metrics.update(mets)

        # Replay
        if self.config.repval_loss:
            feat = sg(repfeat, skip=self.config.repval_grad)
            last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
            boot = imgloss_out['ret'][:, 0].reshape(B, K)
            feat, last, term, rew, boot = jax.tree.map(
                lambda x: x[:, -K:], (feat, last, term, rew, boot))
            inp = self.feat2tensor(feat)
            los, reploss_out, mets = repl_loss(
                last, term, rew, boot,
                self.val(inp, 2),
                self.slowval(inp, 2),
                self.valnorm,
                update=training,
                horizon=self.config.horizon,
                **self.config.repl_loss)
            losses.update(los)
            metrics.update(prefix(mets, 'reploss'))

        assert set(losses.keys()) == set(self.scales.keys()), (
            sorted(losses.keys()), sorted(self.scales.keys()))
        metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
        loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

        carry = (enc_carry, dyn_carry, dec_carry)
        entries = (enc_entries, dyn_entries, dec_entries)
        outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
        return loss, (carry, entries, outs, metrics)

    def report(self, carry, data):
        if not self.config.report:
            return carry, {}

        carry, obs, prevact, _ = self._apply_replay_context(carry, data)
        (enc_carry, dyn_carry, dec_carry) = carry
        B, T = obs['is_first'].shape
        RB = min(6, B)
        metrics = {}

        # Train metrics
        _, (new_carry, entries, outs, mets) = self.loss(
            carry, obs, prevact, training=False)
        mets.update(mets)

        # Grad norms
        if self.config.report_gradnorms:
            for key in self.scales:
                try:
                    lossfn = lambda data, carry: self.loss(
                        carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
                    grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
                    metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
                except KeyError:
                    print(f'Skipping gradnorm summary for missing loss: {key}')

        # Open loop
        firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
        secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
        dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
        dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
        dyn_carry, _, obsfeat = self.dyn.observe(
            dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
            firsthalf(obs['is_first']), training=False)
        _, imgfeat, _ = self.dyn.imagine(
            dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
        dec_carry, _, obsrecons = self.dec(
            dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
        dec_carry, _, imgrecons = self.dec(
            dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
            training=False)

        # Video preds
        for key in self.dec.imgkeys:
            assert obs[key].dtype == jnp.uint8
            true = obs[key][:RB]
            pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
            pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
            error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
            video = jnp.concatenate([true, pred, error], 2)

            video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
            mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
            border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
            border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
            video = jnp.where(mask, video, border[None, :, None, None, :])
            video = jnp.concatenate([video, 0 * video[:, :10]], 1)

            B, T, H, W, C = video.shape
            grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
            metrics[f'openloop/{key}'] = grid

        carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
        return carry, metrics

    def _apply_replay_context(self, carry, data):
        (enc_carry, dyn_carry, dec_carry, prevact) = carry
        carry = (enc_carry, dyn_carry, dec_carry)
        stepid = data['stepid']
        obs = {k: data[k] for k in self.obs_space}
        prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
        prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
        if not self.config.replay_context:
            return carry, obs, prevact, stepid

        K = self.config.replay_context
        nested = elements.tree.nestdict(data)
        entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
        lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
        rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
        rep_carry = (
            self.enc.truncate(lhs(entries[0]), enc_carry),
            self.dyn.truncate(lhs(entries[1]), dyn_carry),
            self.dec.truncate(lhs(entries[2]), dec_carry))
        rep_obs = {k: rhs(data[k]) for k in self.obs_space}
        rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
        rep_stepid = rhs(stepid)

        first_chunk = (data['consec'][:, 0] == 0)
        carry, obs, prevact, stepid = jax.tree.map(
            lambda normal, replay: nn.where(first_chunk, replay, normal),
            (carry, rhs(obs), rhs(prevact), rhs(stepid)),
            (rep_carry, rep_obs, rep_prevact, rep_stepid))
        return carry, obs, prevact, stepid

    def _make_opt(
        self,
        lr: float = 4e-5,
        agc: float = 0.3,
        eps: float = 1e-20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        momentum: bool = True,
        nesterov: bool = False,
        wd: float = 0.0,
        wdregex: str = r'/kernel$',
        schedule: str = 'const',
        warmup: int = 1000,
        anneal: int = 0,
    ):
        chain = []
        chain.append(opt.clip_by_agc(agc))
        chain.append(opt.scale_by_rms(beta2, eps))
        chain.append(opt.scale_by_momentum(beta1, nesterov))
        if wd:
            assert not wdregex[0].isnumeric(), wdregex
            pattern = re.compile(wdregex)
            wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
            chain.append(optax.add_decayed_weights(wd, wdmask))
        assert anneal > 0 or schedule == 'const'
        if schedule == 'const':
            sched = optax.constant_schedule(lr)
        elif schedule == 'linear':
            sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
        elif schedule == 'cosine':
            sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
        else:
            raise NotImplementedError(schedule)
        if warmup:
            ramp = optax.linear_schedule(0.0, lr, warmup)
            sched = optax.join_schedules([ramp, sched], [warmup])
        chain.append(optax.scale_by_learning_rate(sched))
        return optax.chain(*chain)


def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
):
    losses = {}
    metrics = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 if contdisc else 1 - 1 / horizon
    weight = jnp.cumprod(disc * con, 1) / disc
    last = jnp.zeros_like(con)
    term = 1 - con
    ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)

    roffset, rscale = retnorm(ret, update)
    adv = (ret - tarval[:, :-1]) / rscale
    aoffset, ascale = advnorm(adv, update)
    adv_normed = (adv - aoffset) / ascale
    logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
    ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
    policy_loss = sg(weight[:, :-1]) * -(
        logpi * sg(adv_normed) + actent * sum(ents.values()))
    losses['policy'] = policy_loss

    voffset, vscale = valnorm(ret, update)
    tar_normed = (ret - voffset) / vscale
    tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
    losses['value'] = sg(weight[:, :-1]) * (
        value.loss(sg(tar_padded)) +
        slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

    ret_normed = (ret - roffset) / rscale
    metrics['adv'] = adv.mean()
    metrics['adv_std'] = adv.std()
    metrics['adv_mag'] = jnp.abs(adv).mean()
    metrics['rew'] = rew.mean()
    metrics['con'] = con.mean()
    metrics['ret'] = ret_normed.mean()
    metrics['val'] = val.mean()
    metrics['tar'] = tar_normed.mean()
    metrics['weight'] = weight.mean()
    metrics['slowval'] = slowval.mean()
    metrics['ret_min'] = ret_normed.min()
    metrics['ret_max'] = ret_normed.max()
    metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
    for k in act:
        metrics[f'ent/{k}'] = ents[k].mean()
        if hasattr(policy[k], 'minent'):
            lo, hi = policy[k].minent, policy[k].maxent
            metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

    outs = {}
    outs['ret'] = ret
    return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
    losses = {}

    voffset, vscale = valnorm.stats()
    val = value.pred() * vscale + voffset
    slowval = slowvalue.pred() * vscale + voffset
    tarval = slowval if slowtar else val
    disc = 1 - 1 / horizon
    weight = f32(~last)
    ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

    voffset, vscale = valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
    losses['repval'] = weight[:, :-1] * (
        value.loss(sg(ret_padded)) +
        slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

    outs = {}
    outs['ret'] = ret
    metrics = {}

    return losses, outs, metrics


def lambda_return(last, term, rew, val, boot, disc, lam):
    chex.assert_equal_shape((last, term, rew, val, boot))
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc
    cont = (1 - f32(last))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)

