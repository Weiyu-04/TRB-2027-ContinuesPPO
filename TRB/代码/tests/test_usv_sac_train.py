"""
Phase 3 Node B 连续 SAC 训练 setup 冒烟测试（2026-06-17(c)）——独立 SAC 入口接线 + 四方同口径强制 + 变异守护。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_sac_train.py
（自动计数、非 pytest；需 /tmp/trb_T0.xml 夹具，缺则联网下载、离线则端到端块 SKIP。）

覆盖：① make 返回 (SAC, VecNormalize) ② 连续 Box 动作（非 Discrete）③ ent_coef=='auto'（不搬离散 0.01）
④ ent_coef 偷传→ValueError ⑤ colregs_weight 硬编 0.0（probe + 内省）⑥ net [64,64] ⑦ gamma 同进 SAC+VecNorm
⑧ norm_reward 可调（默认 True / False 生效）⑨ assert_continuous_safe_caliber 真 load-bearing（变异翻 raise）
⑩ 端到端 learn（SAC→投影盾→env.step，无 action_masks）⑪ train_continuous_safe 是 Tier3 包装（不在此跑 3M）。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
import gymnasium as gym

from trb_env import usv_sac_train as M
from trb_env.usv_sac_train import (make_continuous_safe_model, assert_continuous_safe_caliber,
                                    train_continuous_safe, CONTINUOUS_SAFE_COLREGS_WEIGHT, POLICY_NET_ARCH,
                                    StableSAC)

_fail = 0
_total = 0


def ok(name, cond):
    global _fail, _total
    _total += 1
    if not cond:
        _fail += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def raises(name, fn, exc=Exception):
    global _fail, _total
    _total += 1
    try:
        fn()
        print(f"[FAIL] {name}（未抛异常）")
        _fail += 1
    except exc:
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}（抛了 {type(e).__name__} 而非 {exc.__name__}: {e}）")
        _fail += 1


# ---- 不依赖场景的静态守护（即便离线也跑）----
ok("A 模块常量 CONTINUOUS_SAFE_COLREGS_WEIGHT==0.0（丢 r_colregs 硬编，D37-B）",
   CONTINUOUS_SAFE_COLREGS_WEIGHT == 0.0)
ok("B POLICY_NET_ARCH 单一真相源==[64,64]（与离散同网络、忠实 Krasowski）",
   list(POLICY_NET_ARCH) == [64, 64])

# ---- ⑫f/⑫g L65 StableSAC.train() 忠实性 CI（源码级·无需场景夹具·把人工核对做成永久回归）----
import inspect as _inspect, difflib as _difflib
def _clean(src):                                                  # 去行尾/整行注释 + 空行 → 规整行序列
    return [s.strip() for s in (ln.split("#", 1)[0].rstrip() for ln in src.splitlines()) if s.strip()]
# StableSAC.train 相对 sb3 2.3.2 SAC.train 允许的【唯一】新增行（★L65 梯度裁剪 4 行 + ★L67 target-Q 裁剪 2 行）：
_EXPECTED_INSERTS = {
    "if self.max_grad_norm is not None:",
    "th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm,",
    "th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm,",
    "error_if_nonfinite=True)",
    "if self.target_q_clip is not None:",
    "target_q_values = th.clamp(target_q_values, -self.target_q_clip, self.target_q_clip)",
}
_stable_src = _inspect.getsource(StableSAC.train)
_sb3_lines = _clean(_inspect.getsource(SAC.train))
_stable_lines = _clean(_stable_src)
ok("⑫f StableSAC.train 两处梯度裁剪均带 error_if_nonfinite=True（NaN/inf 梯度 fail-fast·B5g·别静默传染烧算力）",
   _stable_src.count("clip_grad_norm_") == 2 and _stable_src.count("error_if_nonfinite=True") == 2)
ok("⑫h StableSAC.train 含 target-Q 值裁剪行（★L67·th.clamp(target_q_values…)·对症修法被移除即 FAIL）",
   _stable_src.count("th.clamp(target_q_values") == 1 and "self.target_q_clip is not None" in _stable_src)
# ⑫g【难绕过】忠实性 CI（红队 B②）：SequenceMatcher 断言 StableSAC.train 相对 sb3【仅插入白名单裁剪/clamp 行·未改/删任何 sb3 原逻辑】
# ——裸子串剥离会被"含 token 的 sabotage 行"绕过（如 `self.gamma=0.0 if self.target_q_clip is None else ...`）；本版任何改/删 sb3 原行 或 插非白名单行都暴露。
_bad = []
for _tag, _i1, _i2, _j1, _j2 in _difflib.SequenceMatcher(a=_sb3_lines, b=_stable_lines, autojunk=False).get_opcodes():
    if _tag == "equal":
        continue
    if _tag == "insert":
        _bad += [("插入非白名单行", ln) for ln in _stable_lines[_j1:_j2] if ln not in _EXPECTED_INSERTS]
    else:                                                          # replace/delete = 动了 sb3 原逻辑行（含白名单 token 的 sabotage 在此暴露）
        _bad += [(_tag + "·改删 sb3 原行", ln) for ln in _sb3_lines[_i1:_i2]]
        _bad += [(_tag + "·引入非白名单行", ln) for ln in _stable_lines[_j1:_j2] if ln not in _EXPECTED_INSERTS]
if _bad:
    print("    ⑫g 异常行（应空）:", _bad[:4])
ok("⑫g StableSAC.train 相对 sb3 2.3.2 SAC.train【仅插入白名单裁剪/clamp 行·未改删任何 sb3 原逻辑】（忠实性 CI·难绕过·红队 B②）",
   not _bad)

_T0 = "/tmp/trb_T0.xml"
_HAVE_T0 = os.path.exists(_T0)
if not _HAVE_T0:
    try:
        import urllib.request
        url = ("https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
               "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml")
        urllib.request.urlretrieve(url, _T0)
        _HAVE_T0 = True
    except Exception as e:
        print(f"[SKIP] /tmp/trb_T0.xml 不在且离线下载失败（{e}）→ 跳过端到端块（非代码回归）")

if _HAVE_T0:
    from commonocean.common.file_reader import CommonOceanFileReader
    _sc, _pp = CommonOceanFileReader(_T0).open()
    _ppx = list(_pp.planning_problem_dict.values())[0]
    _POOL = [(_sc, _ppx)]

    # 小 buffer 省内存（SAC 默认 buffer_size=1M 会一次性分配大块）；学习极少步只为验集成
    _SK = dict(buffer_size=5000, learning_starts=20, batch_size=32)

    def _make(**kw):
        return make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, **{**_SK, **kw})

    model, venv = _make()

    # ① 类型 + ② 连续 Box 动作（非 Discrete）
    ok("① make 返回 (SAC, VecNormalize)", isinstance(model, SAC) and isinstance(venv, VecNormalize))
    ok("② 连续动作空间 = Box(2,)（非 Discrete；连续臂坐实）",
       isinstance(venv.action_space, gym.spaces.Box) and venv.action_space.shape == (2,))

    # ③ ent_coef 自带最大熵（不搬离散 0.01，D38）
    ok("③ SAC ent_coef=='auto'（自带最大熵未被覆盖，D38）", str(model.ent_coef) == "auto")

    # ⑤ colregs_weight 硬编 0.0（内省实际构造的 env）
    _cw = float(venv.venv.envs[0]._inner.env.reward_fn.colregs_weight)
    ok("⑤ 构造的 env colregs_weight==0.0（丢 r_colregs，D37-B；堵 D40 footgun）", _cw == 0.0)

    # ⑥ net_arch [64,64]
    ok("⑥ policy net_arch==[64,64]（忠实 Krasowski §VII）",
       list(model.policy_kwargs.get("net_arch")) == [64, 64])

    # ⑦ gamma 同进 SAC 与 VecNormalize（L21 MINOR②）
    ok("⑦ gamma 同进 SAC(0.99) 与 VecNormalize(0.99)（一致，L21 MINOR②）",
       abs(float(model.gamma) - float(venv.gamma)) < 1e-12)

    # ⑧ norm_reward 默认 True / 可调 False
    ok("⑧a 默认 norm_reward==True（依 D38 四方平价）", bool(venv.norm_reward) is True)
    _m2, _v2 = _make(norm_reward=False)
    ok("⑧b norm_reward=False 生效（off-policy 张力时可关，钱图指标不受影响）", bool(_v2.norm_reward) is False)
    ok("⑧c norm_obs 恒 True（obs 归一化是停船配方命门 D22、不可关）", bool(_v2.norm_obs) is True)

    # ⑮ L76：SAC gradient_steps（UTD 控制·速度/学习动态旋钮）经 sac_kwargs 透传落地 + 不破 ent_coef='auto'（D38）
    ok("⑮a gradient_steps 默认 == sb3 原生 1（不传=现状不变·n_envs=1 时 UTD=1）", int(model.gradient_steps) == 1)
    _mgs, _vgs = _make(gradient_steps=16)
    ok("⑮b gradient_steps=16 经 sac_kwargs 透传【真落地】model.gradient_steps==16（L76 速度/UTD 旋钮·变异硬编即 FAIL）",
       int(_mgs.gradient_steps) == 16 and str(_mgs.ent_coef) == "auto")   # 同验 D38 熵守卫不被本旋钮破

    # ④ ent_coef / target_entropy 偷传 → ValueError（防把离散 0.01 搬进来 + 自带最大熵机制不接受外覆）
    raises("④a 偷传 ent_coef=0.01 → ValueError（D38 守护，别搬离散熵）",
           lambda: _make(ent_coef=0.01), ValueError)
    raises("④b 偷传 target_entropy=0.0 → ValueError（红队 LOW：自带最大熵目标熵不接受外覆）",
           lambda: _make(target_entropy=0.0), ValueError)

    # ⑨ assert_continuous_safe_caliber：正常通过 + 变异翻 raise（真 load-bearing）
    ok("⑨a 正常模型通过口径自检", assert_continuous_safe_caliber(model, venv) is None)

    def _mut_gamma():
        m3, v3 = _make()
        m3.gamma = 0.95          # 制造 SAC↔VecNorm gamma 不一致
        assert_continuous_safe_caliber(m3, v3)
    raises("⑨b 变异守护：gamma 不一致 → 口径自检 raise", _mut_gamma, AssertionError)

    def _mut_reward_gamma():
        m3b, v3b = _make()
        for _e in v3b.venv.get_attr("env"):   # 仅篡改 reward_fn(修法A PBRS)gamma·model/VecNorm 仍 0.99 → VecNorm-vs-model 检查通过、唯靠新 reward_fn.gamma 检查抓（L82）
            _e.reward_fn.gamma = 0.95
        assert_continuous_safe_caliber(m3b, v3b)
    raises("⑨b2 变异守护：reward_fn.gamma 漂移(model/VecNorm 仍同源) → 口径自检 raise（修法A PBRS gamma 第三处同源）", _mut_reward_gamma, AssertionError)

    def _mut_cw():
        m4, v4 = _make()
        v4.venv.envs[0]._inner.env.reward_fn.colregs_weight = 1.0   # 制造 r_colregs 被复活
        assert_continuous_safe_caliber(m4, v4)
    raises("⑨c 变异守护：colregs_weight 被复活 1.0 → 口径自检 raise", _mut_cw, AssertionError)

    def _mut_novecnorm():
        m5, v5 = _make(use_vecnorm=False)   # 无 VecNormalize（make 不自调 caliber，由此处显式调）
        assert_continuous_safe_caliber(m5, v5)
    raises("⑨d 变异守护：无 VecNormalize → 口径自检 raise", _mut_novecnorm, AssertionError)

    # ⑨e make【构造时自调 caliber】堵 net_arch 静默覆盖（双 agent MEDIUM）：policy_kwargs 覆盖 net_arch → make 即 raise
    raises("⑨e 防御做在前：偷传 net_arch=[32,32] → make 构造时口径自检即 AssertionError（不靠调用方记得调）",
           lambda: _make(policy_kwargs=dict(net_arch=[32, 32])), AssertionError)

    # ⑨f 变异守护：clip_obs 被篡改 → 口径自检 raise（obs 归一化口径完整性，红队 LOW）
    def _mut_clipobs():
        m8, v8 = _make()
        v8.clip_obs = 999.0
        assert_continuous_safe_caliber(m8, v8)
    raises("⑨f 变异守护：clip_obs 被篡改 999 → 口径自检 raise（obs 平价完整）", _mut_clipobs, AssertionError)

    # ⑩ 端到端 learn（SAC 连续动作 → 投影盾 safe_action → env.step；无 action_masks）；强断言 num_timesteps 真推进
    def _learn():
        m6, _ = _make()
        m6.learn(total_timesteps=80, log_interval=None)   # >learning_starts(20) 触发梯度步
        return int(m6.num_timesteps)
    ok("⑩ 端到端 learn(80) 跑通且 num_timesteps≥80 真推进（SAC→投影盾→env.step、无 action_masks）",
       _learn() >= 80)

    # ⑪ train_continuous_safe 存在且是 Tier3 包装（此处仅极小步验集成、非 3M）
    def _tier3_smoke():
        m7, v7 = train_continuous_safe(scenario_pool=_POOL, n_envs=1, seed=0,
                                       total_timesteps=40, **_SK)
        return isinstance(m7, SAC)
    ok("⑪ train_continuous_safe(40) 集成跑通（真训练 3M=Tier3 待 user）", _tier3_smoke())

    # ⑫ L65 Fix：StableSAC 梯度裁剪（修 critic Q 高估发散）
    ok("⑫ make 返回 StableSAC(=SAC+梯度裁剪·L65) + max_grad_norm=1.0 默认 + lr=1e-4",
       isinstance(model, StableSAC) and model.max_grad_norm == 1.0 and abs(model.learning_rate - 1e-4) < 1e-12)
    def _mut_vanilla():                                    # 变异：vanilla SAC（无梯度裁剪）→ caliber 应 raise
        from stable_baselines3 import SAC as _VanillaSAC
        mv = _VanillaSAC("MlpPolicy", venv, policy_kwargs=dict(net_arch=POLICY_NET_ARCH), gamma=0.99, seed=0)
        assert_continuous_safe_caliber(mv, venv)
    raises("⑫b 变异守护：vanilla SAC(无梯度裁剪) → 口径自检 raise（L65 须 StableSAC·防回归丢裁剪=发散）",
           _mut_vanilla, AssertionError)
    def _crit_delta(mgn):                                  # critic 参数训练后总变化（裁剪生效则 tiny<<none）
        import torch as _th
        mm, vv = make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, learning_starts=10, max_grad_norm=mgn)
        p0 = [p.detach().clone() for p in mm.critic.parameters()]
        mm.learn(total_timesteps=40)
        d = sum(float((p - q).abs().sum()) for p, q in zip(mm.critic.parameters(), p0)); vv.close(); return d
    ok("⑫c 梯度裁剪【真生效】：tiny-clip(1e-6) 的 critic 参数变化 << no-clip(None)（功能变异坐实裁剪非空操作）",
       _crit_delta(1e-6) < _crit_delta(None) * 0.5)
    mN, vN = make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, max_grad_norm=None)
    ok("⑫d max_grad_norm=None → 退化(仍 StableSAC·不裁剪·A/B 基线复现发散用)",
       isinstance(mN, StableSAC) and mN.max_grad_norm is None); vN.close()
    import trb_env.usv_sac_train as _ust                   # ⑫e 版本断言守护：防 sb3 升级后用 stale train() 复制
    _ov = _ust._sb3.__version__
    try:
        _ust._sb3.__version__ = "9.9.9"
        raises("⑫e 版本断言守护：sb3≠2.3.2 → StableSAC 构造 RuntimeError（防 stale train() 复制丢新逻辑）",
               lambda: StableSAC("MlpPolicy", venv), RuntimeError)
    finally:
        _ust._sb3.__version__ = _ov

    # ⑫i-k L67 Fix：target-Q 值裁剪（对症修 critic Q 高估发散·正确作用面）
    ok("⑫i make 默认 target_q_clip=None（保 L65 行为·A/B 对照）+ 显式值穿透到 model.target_q_clip",
       model.target_q_clip is None and StableSAC("MlpPolicy", venv, target_q_clip=37.0).target_q_clip == 37.0)
    raises("⑫j 变异守护：target_q_clip<=0 → __init__ ValueError（Q 裁剪天花板须有限正数）",
           lambda: StableSAC("MlpPolicy", venv, target_q_clip=0.0), ValueError)
    raises("⑫j2 变异守护：target_q_clip=nan → __init__ ValueError（防 nan<=0 漏过=静默 NaN 污染·红队 B5）",
           lambda: StableSAC("MlpPolicy", venv, target_q_clip=float("nan")), ValueError)
    raises("⑫j3 变异守护：target_q_clip=inf → __init__ ValueError（非有限值拒收）",
           lambda: StableSAC("MlpPolicy", venv, target_q_clip=float("inf")), ValueError)
    def _maxq(tqc):                                       # 训后 critic 在【固定确定性】一批 obs 上的 max|Q|
        import torch as _th
        mm, vv = make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0,
                                             learning_starts=10, max_grad_norm=None, target_q_clip=tqc)
        mm.learn(total_timesteps=120)
        _g = _th.Generator().manual_seed(0)               # 确定性读出（别用 space.sample 的非种子 RNG·防 flaky）
        obs = _th.randn((8, vv.observation_space.shape[0]), generator=_g)
        act = _th.zeros((8, vv.action_space.shape[0]))
        with _th.no_grad():
            q = _th.cat(mm.critic(obs, act), dim=1).abs().max().item()
        vv.close(); return q
    _q_clip, _q_none = _maxq(0.05), _maxq(None)           # 差分（红队 ④：绝对阈值<2.0 假守护·去 clamp=no-op→两者相等仍 PASS）
    ok(f"⑫k target-Q 裁剪【真 cap Q·差分坐实·红队④】：极紧 clip(0.05) 的 max|Q|={_q_clip:.3f} 显著 < 无裁剪(None)={_q_none:.3f}（去 clamp=no-op→两者相等→FAIL）",
       _q_clip < _q_none * 0.5 and _q_clip < 0.3)

    # ⑬ L67-续2：LayerNorm critic（根因修 Q 高估发散）+ n_critics(REDQ-lite) + tau knob
    import torch as _th
    from trb_env.usv_sac_train import LayerNormSACPolicy, _critic_has_layernorm
    _m_ln, _v_ln = make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, learning_starts=10,
                                              use_critic_layernorm=True, n_critics=5, tau=0.01)
    _n_ln = sum(1 for mod in _m_ln.critic.modules() if isinstance(mod, _th.nn.LayerNorm))
    ok("⑬a use_critic_layernorm=True → critic 真含 LayerNorm + policy=LayerNormSACPolicy（接线坐实）",
       _critic_has_layernorm(_m_ln) and isinstance(_m_ln.policy, LayerNormSACPolicy))
    ok("⑬b n_critics=5 × 每 critic 2 隐层 → 10 个 LayerNorm + critic.n_critics==5（REDQ-lite knob 穿透）",
       _n_ln == 10 and _m_ln.critic.n_critics == 5)
    ok("⑬c tau knob 穿透（model.tau==0.01·默认 0.005）", abs(_m_ln.tau - 0.01) < 1e-12)
    ok("⑬d 默认 use_critic_layernorm=False → vanilla MlpPolicy·critic 无 LayerNorm（A/B 基线·additive 不变）",
       not _critic_has_layernorm(model) and not isinstance(model.policy, LayerNormSACPolicy))
    def _mut_ln_partial():                               # 变异：只 qf1 换无 LayerNorm（部分畸形态）→ all() 强谓词 False → caliber raise（红队②强化）
        _q1 = _m_ln.critic.qf1
        try:
            _m_ln.critic.qf1 = _th.nn.Sequential(_th.nn.Linear(2, 1))
            _m_ln.critic.q_networks = [getattr(_m_ln.critic, f"qf{i}") for i in range(_m_ln.critic.n_critics)]
            assert_continuous_safe_caliber(_m_ln, _v_ln)
        finally:
            _m_ln.critic.qf1 = _q1
            _m_ln.critic.q_networks = [getattr(_m_ln.critic, f"qf{i}") for i in range(_m_ln.critic.n_critics)]
    raises("⑬e caliber 一致性守护（强 all() 谓词·红队②）：仅 1 个 critic 失 LayerNorm（部分畸形）→ AssertionError",
           _mut_ln_partial, AssertionError)
    _v_ln.close()
    raises("⑬g 变异守护：tau 越界(1.5) → ValueError（红队③·sb3 不校验 tau·docstring 既宣传为 knob 就该守）",
           lambda: make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, tau=1.5), ValueError)
    raises("⑬h 变异守护：n_critics<1(0) → ValueError",
           lambda: make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, n_critics=0), ValueError)
    def _maxq_arch(ln, steps=300):                       # 高 lr(3e-4=之前"炸"档) 训后 max|Q|：LayerNorm 应更压
        mm, vv = make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, learning_starts=10,
                                            max_grad_norm=None, learning_rate=3e-4, use_critic_layernorm=ln)
        mm.learn(total_timesteps=steps)
        g = _th.Generator().manual_seed(1)
        obs = _th.randn((16, vv.observation_space.shape[0]), generator=g)
        act = _th.zeros((16, vv.action_space.shape[0]))
        with _th.no_grad():
            q = _th.cat(mm.critic(obs, act), dim=1).abs().max().item()
        vv.close(); return q
    _q_ln_arch, _q_no_arch = _maxq_arch(True), _maxq_arch(False)
    ok(f"⑬f LayerNorm critic【真压 Q·根因生效】：高 lr(3e-4) 训后 max|Q|={_q_ln_arch:.2f} < vanilla={_q_no_arch:.2f}（LayerNorm 接错=回退普通 MLP→两者相等→FAIL）",
       _q_ln_arch < _q_no_arch)
    # ⑬i/j L67-续7：完整 BRO = LayerNorm + critic AdamW 解耦权重衰减（闭深核 LN-5 权重驱动残余发散通道）
    _m_wd, _v_wd = make_continuous_safe_model(scenario_pool=_POOL, n_envs=1, seed=0, learning_starts=10,
                                              use_critic_layernorm=True, critic_weight_decay=1e-3)
    _opt = _m_wd.critic.optimizer
    _dec = _opt.param_groups[0]; _nodec = _opt.param_groups[1]
    ok("⑬i critic_weight_decay>0 → critic 优化器 AdamW + 【解耦】(衰减组只含 Linear 权重 ndim≥2·不衰减组含 bias/LN affine ndim≤1)·actor 不动 Adam",
       type(_opt).__name__ == "AdamW" and abs(_dec["weight_decay"] - 1e-3) < 1e-12 and _nodec["weight_decay"] == 0.0
       and all(p.ndim >= 2 for p in _dec["params"]) and all(p.ndim <= 1 for p in _nodec["params"])
       and type(_m_wd.actor.optimizer).__name__ == "Adam")
    ok("⑬j 默认 critic_weight_decay=0 → critic 仍 Adam（A/B 基线·纯 LayerNorm 不变·additive）",
       type(_m_ln.critic.optimizer).__name__ == "Adam")
    # ⑬k L67-续8（二审 BRO-3 MAJOR）：完整 BRO(critic AdamW 2 param_group) checkpoint·裸 SAC.load 崩(组数不匹配)·load_sac_for_eval 修
    import os as _os, tempfile as _tf2
    from trb_env.usv_sac_train import load_sac_for_eval
    from stable_baselines3 import SAC as _SAC
    _m_wd.learn(total_timesteps=20)
    _ckp = _os.path.join(_tf2.mkdtemp(), "bro.zip"); _m_wd.save(_ckp)
    _bare_crashed = False
    try:
        _SAC.load(_ckp, device="cpu")
    except ValueError:
        _bare_crashed = True                                # 裸 SAC.load 对 wd>0(AdamW 2组)必崩=BRO-3
    _g = np.random.default_rng(2); _obs = _g.standard_normal((8, _v_wd.observation_space.shape[0])).astype("float32")
    _m_rl = load_sac_for_eval(_ckp, device="cpu")           # 鲁棒 loader：跳优化器·只灌 policy
    _a1, _ = _m_wd.predict(_obs, deterministic=True); _a2, _ = _m_rl.predict(_obs, deterministic=True)
    ok("⑬k 完整 BRO checkpoint replay 鲁棒（二审 BRO-3）：裸 SAC.load 崩(AdamW 2组≠默认 Adam 1组)·load_sac_for_eval 成功+predict 逐位一致",
       _bare_crashed and np.array_equal(_a1, _a2))
    _v_wd.close()

    # ⑭ 节点2/L67-续3：连续 PPO 入口（SAC 岔路的并行 hedge·on-policy 无 off-policy 发散·四方全 PPO 撤算法混淆）
    from trb_env.usv_sac_train import make_continuous_safe_ppo_model, assert_continuous_safe_ppo_caliber
    from stable_baselines3 import PPO as _PPO
    from trb_env.train import ENT_COEF as _ENT, VECNORM_KWARGS
    _mp, _vp = make_continuous_safe_ppo_model(scenario_pool=_POOL, n_envs=1, seed=0, subproc=False,
                                              n_steps=128, batch_size=64)
    _hi = _vp.action_space.high
    ok("⑭a make_continuous_safe_ppo_model → sb3 PPO（非 MaskablePPO）+ 动作箱 ±0.048/±0.018 + ent_coef=离散同款 + net[64,64] + gamma0.99",
       isinstance(_mp, _PPO) and abs(float(_hi[0]) - 0.048) < 1e-9 and abs(float(_hi[1]) - 0.018) < 1e-9
       and abs(_mp.ent_coef - _ENT) < 1e-12 and list(_mp.policy_kwargs.get("net_arch")) == [64, 64]
       and abs(_mp.gamma - 0.99) < 1e-12)
    ok("⑭b PPO 臂 colregs_weight 硬编 0.0（丢 r_colregs·四方平价）+ VecNormalize norm_obs/clip_obs 同款",
       all(float(e.reward_fn.colregs_weight) == 0.0 for e in _vp.venv.get_attr("env"))
       and _vp.norm_obs and float(_vp.clip_obs) == float(VECNORM_KWARGS["clip_obs"]))
    def _mut_ppo_colregs():                              # 变异：colregs 复活 → PPO caliber raise
        _saved = [e.reward_fn.colregs_weight for e in _vp.venv.get_attr("env")]
        try:
            for e in _vp.venv.get_attr("env"):
                e.reward_fn.colregs_weight = 1.0
            assert_continuous_safe_ppo_caliber(_mp, _vp)
        finally:
            for e, c in zip(_vp.venv.get_attr("env"), _saved):
                e.reward_fn.colregs_weight = c
    raises("⑭c PPO caliber 变异守护：colregs_weight 被复活 1.0 → AssertionError（破四方平价）",
           _mut_ppo_colregs, AssertionError)
    def _mut_ppo_algo():                                 # 变异：把 SAC 模型喂 PPO caliber → raise（算法须 sb3 PPO）
        assert_continuous_safe_ppo_caliber(model, _vp)
    raises("⑭d PPO caliber 变异守护：非 PPO 模型(SAC) → AssertionError（连续 PPO 臂算法须 sb3 PPO）",
           _mut_ppo_algo, AssertionError)
    # ⑭f/g 深核 F1（MAJOR）：sb3 PPO 高斯不 squash·log_std_init=0→σ=1.0 远超动作箱 ±0.048/±0.018→采样全 clip 到边界=探索退化。
    _init_std = np.exp(_mp.policy.log_std.detach().numpy())
    ok(f"⑭f F1 修复：PPO 初始高斯 σ={np.round(_init_std,4)} 落进动作箱内(≤±0.048/±0.018·非 σ=1 退化探索·log_std_init 从箱派生)",
       bool(np.all(_init_std <= np.array([0.048, 0.018]) + 1e-9)))
    raises("⑭g F1 caliber 守护：log_std_init=0(σ=1)→超动作箱→AssertionError（拦 std=1 全 clip footgun·深核 MAJOR）",
           lambda: make_continuous_safe_ppo_model(scenario_pool=_POOL, n_envs=1, seed=0, subproc=False, log_std_init=0.0),
           AssertionError)
    _mp.learn(total_timesteps=256)
    ok("⑭e 连续 PPO 端到端 learn(256) 跑通（PPO→投影盾→env.step·on-policy·num_timesteps≥256）",
       _mp.num_timesteps >= 256)
    _vp.close()

# ---------------- ⑮ LR 退火 schedule + sync callback（`03` L88·分段鲁棒·save/load 保住）----------------
# 用 CartPole+PPO（不依赖 commonocean·始终跑）·复刻 run_step4e 分段 learn(reset_num_timesteps=(c==0)) 验承重机制。
import warnings as _warnings
_warnings.filterwarnings("ignore")
from stable_baselines3 import PPO as _PPO_LR
import tempfile as _tempfile

_Sch = M.LRAnnealSchedule
# ⑮a 线性退火语义 + 端点 clamp
_s = _Sch(3e-4, 0.0, 1000)
_s.num_timesteps = 0;    _v0 = _s(1.0)
_s.num_timesteps = 500;  _v5 = _s(0.5)
_s.num_timesteps = 1000; _v1 = _s(0.0)
_s.num_timesteps = 2000; _vc = _s(0.0)   # 超 anneal_steps → clamp 恒 end
ok("⑮a LRAnnealSchedule 线性 start→end + 端点 clamp（3e-4→1.5e-4→0→0）",
   abs(_v0 - 3e-4) < 1e-12 and abs(_v5 - 1.5e-4) < 1e-12 and abs(_v1) < 1e-12 and abs(_vc) < 1e-12)
# ⑮b 有意忽略 SB3 传入的 progress_remaining（同 num_timesteps → 同 lr·防分段锯齿）
_s.num_timesteps = 500
ok("⑮b LRAnnealSchedule 忽略 progress_remaining（分段锯齿免疫·读累积 num_timesteps）",
   _s(1.0) == _s(0.0) == _s(0.37))


def _seg_lr_trace(install_anneal, total=256, n_seg=4):
    """复刻 run_step4e 分段训练：返回每段末 optimizer lr。install_anneal=True 则装 LRAnnealSchedule+SyncCallback。"""
    m = _PPO_LR("MlpPolicy", "CartPole-v1", n_steps=64, batch_size=64, n_epochs=2, seed=0, verbose=0, device="cpu")
    sched = None
    if install_anneal:
        sched = _Sch(float(m.lr_schedule(1.0)), 0.0, total)
        m.learning_rate = sched
        m.lr_schedule = sched
    seg = total // n_seg
    lrs = []
    for c in range(n_seg):
        cb = M.LRAnnealSyncCallback(sched) if sched is not None else None
        m.learn(total_timesteps=seg, reset_num_timesteps=(c == 0), callback=cb)
        lrs.append(m.policy.optimizer.param_groups[0]["lr"])
    return m, lrs, sched

# ⑮c 分段训练下单调退火（核心机制·替 SB3 每段归零锯齿）
_m_on, _lrs_on, _sch_on = _seg_lr_trace(True)
ok("⑮c 分段 learn(reset_num_timesteps=c==0) 下 LR 单调不增退火且到端点≈0（替 SB3 锯齿 progress·全局累积步）",
   all(_lrs_on[i] >= _lrs_on[i + 1] - 1e-12 for i in range(len(_lrs_on) - 1))
   and _lrs_on[0] > 1e-4 and _lrs_on[-1] < 1e-5 and _sch_on.num_timesteps == 256)
# ⑮d OFF 路径（不装退火）optimizer lr 全程恒 3e-4（=训练字节级不变前提·与 ON 隔离）
_m_off, _lrs_off, _ = _seg_lr_trace(False)
ok("⑮d OFF 路径 optimizer lr 全程恒 3e-4（不装退火=训练字节级不变前提）",
   all(abs(x - 3e-4) < 1e-12 for x in _lrs_off))
# ⑮e save/load 保住退火（learning_rate 复活为 LRAnnealSchedule·lr_schedule 值仍退火态·非重置回 3e-4）
with _tempfile.TemporaryDirectory() as _d:
    _pp = os.path.join(_d, "m.zip")
    _m_on.save(_pp)
    _m2 = _PPO_LR.load(_pp, device="cpu")
    _reload_lr = _m2.lr_schedule(0.5)
    ok("⑮e save/load 保住退火：load 后 lr_schedule 返回退火态(≈0·非重置 3e-4) + learning_rate 复活 LRAnnealSchedule（Layer-2 可重挂同步）",
       _reload_lr < 1e-5 and type(_m2.learning_rate).__name__ == "LRAnnealSchedule"
       and getattr(_m2.learning_rate, "num_timesteps", -1) == 256)
    # ⑮f load 后 eval predict 可用（checkpoint 主用途=replay_eval·不训练→lr_schedule 不被调·退火无害）
    _a, _ = _m2.predict(np.zeros((1, 4), dtype=np.float32), deterministic=True)
    ok("⑮f load 后 predict 可用（退火 schedule 不破坏 eval/replay 路径）", _a is not None)

# ---------------- P1(L147 复审补测) PursuitNaivePolicy 朴素基线（未 bind→raise / bound→动作∈箱+几何合理 / SB3 无 bind_env 钩子 no-op·此前 0 committed 用例）----------------
from trb_env.usv_sac_train import PursuitNaivePolicy
from trb_env.usv_colregs import VesselState
import trb_env.evaluate as _EV
import inspect as _inspect

# ① 未 bind_env → predict raise RuntimeError（防裸用漏 bind 静默乱走）
_pn = PursuitNaivePolicy(dt=10.0, v_max=9.5, a_box=0.048, w_box=0.018)
_p1_raised = False
try:
    _pn.predict(np.zeros(27, dtype=np.float32))
except RuntimeError:
    _p1_raised = True
except Exception:
    _p1_raised = False
ok("P1 未 bind_env → predict raise RuntimeError（防漏 bind 静默乱走）", _p1_raised)


class _FakeInner:                                      # duck-type：run_episode_continuous 读 env.env.goal_center
    def __init__(self, goal):
        self.goal_center = np.asarray(goal, float)


class _FakeEnv:                                        # duck-type：_ego_vs()/_obs_vs()/env（PursuitNaivePolicy 只读这三样原始几何）
    def __init__(self, ego, goal, obs=None):
        self._ego = ego
        self.env = _FakeInner(goal)
        self._obs = obs

    def _ego_vs(self):
        return self._ego

    def _obs_vs(self):
        return self._obs


_ego = VesselState(position=np.array([0.0, 0.0]), orientation=0.0, velocity=9.5, length=175.0)
# ② 目标正前+无他船 → 直行全速（head_err≈0→ω≈0；开阔→accel≈0）·动作 ∈ RL 箱
_pn.bind_env(_FakeEnv(_ego, goal=[5000.0, 0.0]))
_afwd, _ = _pn.predict(np.zeros(27, dtype=np.float32))
ok(f"P1 动作 ∈ RL 箱 ±0.048/±0.018（{_afwd}）",
   abs(float(_afwd[0])) <= 0.048 + 1e-6 and abs(float(_afwd[1])) <= 0.018 + 1e-6)
ok(f"P1 目标正前+无他船 → ω≈0 & accel≈0（开阔全速直行·{_afwd}）",
   abs(float(_afwd[1])) < 1e-5 and abs(float(_afwd[0])) < 1e-5)
# ③ 目标正后 → ω 饱和转向（真追目标·非乱转/稻草人）
_pn.bind_env(_FakeEnv(_ego, goal=[-5000.0, 0.0]))
_aback, _ = _pn.predict(np.zeros(27, dtype=np.float32))
ok(f"P1 目标正后 → |ω| 饱和到 w_box=0.018（真追目标非稻草人·{_aback}）",
   abs(abs(float(_aback[1])) - 0.018) < 1e-6)
# ④ 他船近场(<d_slow=3000) → 减速让盾余量（v_target=v_slow<v_ego → accel<0）
_obs_near = VesselState(position=np.array([500.0, 0.0]), orientation=0.0, velocity=9.5, length=175.0)
_pn.bind_env(_FakeEnv(_ego, goal=[5000.0, 0.0], obs=_obs_near))
_aslow, _ = _pn.predict(np.zeros(27, dtype=np.float32))
ok(f"P1 他船近场(<d_slow) → 减速 accel<0（给盾余量·{_aslow}）", float(_aslow[0]) < 0.0)
# ⑤b 转向【方向/符号】锁（偏轴目标·变异审 HIGH 补·L147）：②③ 的目标落在本船艏艉轴上→ω 符号不可见→"追反了"(转离目标)的伪策略能骗过它们；
#     偏轴目标必须锁符号，否则 why-RL 消融 B 臂"更强/更公平基线"可信度破产（基线其实追反·RL 赢它无意义）。
_pn.bind_env(_FakeEnv(_ego, goal=[0.0, 5000.0]))       # 正左(+y·艏向 0→需 +90°) → 应【左转】ω>0
_aL, _ = _pn.predict(np.zeros(27, dtype=np.float32))
_pn.bind_env(_FakeEnv(_ego, goal=[0.0, -5000.0]))      # 正右(−y) → 应【右转】ω<0
_aR, _ = _pn.predict(np.zeros(27, dtype=np.float32))
ok(f"P1 偏轴目标转向方向正确：正左→ω>0({float(_aL[1]):.3f}) & 正右→ω<0({float(_aR[1]):.3f})（锁 ω 符号·防追反）",
   float(_aL[1]) > 1e-3 and float(_aR[1]) < -1e-3)
# ⑤ evaluate.run_episode_continuous 的 bind 钩子契约：hasattr(model,'bind_env') 守卫 → SB3 无此方法→跳过(RL bit-identical)·朴素有
_ev_src = _inspect.getsource(_EV.run_episode_continuous)
ok("P1 run_episode_continuous 用 hasattr(model,'bind_env') 守卫（SB3 无此→跳过→RL 路径 bit-identical）",
   'hasattr(model, "bind_env")' in _ev_src or "hasattr(model, 'bind_env')" in _ev_src)
ok("P1 PursuitNaivePolicy 有 bind_env / SB3 SAC 无（钩子只对朴素基线生效·不误绑 RL）",
   hasattr(_pn, "bind_env") and not hasattr(SAC, "bind_env"))


print()
if _fail == 0:
    if _HAVE_T0:
        print(f"✅ 全部 PASS（{_total} 项，含端到端）")
    else:
        print(f"⚠️ 仅静态 {_total} 项 PASS — 端到端块 SKIP（无 T-0 夹具且离线）；勿当完整通过（红队 MINOR：诚实计数）")
else:
    print(f"❌ {_fail}/{_total} FAIL")
    sys.exit(1)
