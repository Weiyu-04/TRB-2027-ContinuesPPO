"""
SE-RL 安全盾层冒烟测试（step4c）—— mask 各档正确性 + 49 槽不变量 + As=∅ 兜底 + a_em 传递 + 端到端。
跑：/opt/miniconda3/envs/trb/bin/python -B 代码/tests/test_usv_shield.py
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from commonocean.common.file_reader import CommonOceanFileReader
from trb_env.usv_shield import ShieldedUSVEnv, ACTION_TO_IDX, _AKEEP_IDX, _key
from trb_env.usv_env import DISCRETE_ACTIONS, IDX_EMERGENCY, N_ACTIONS_TOTAL
from trb_env.usv_colregs import (A_KEEP, ATR, RHO_NO_CONFLICT, RHO_STAND_ON, RHO_HEAD_ON,
                                 RHO_CROSSING, RHO_OVERTAKE, RHO_EMERGENCY)

_fail = 0
def check(name, ok):
    global _fail
    if not ok: _fail += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

_T0 = "/tmp/trb_T0.xml"
if not os.path.exists(_T0):
    import urllib.request
    url = ("https://gitlab.lrz.de/tum-cps/commonocean-scenarios/-/raw/main/scenarios/"
           "HandcraftedTwoVesselEncounters_01_24/ZAM_AAA-1_20240121_T-0.xml")
    urllib.request.urlretrieve(url, _T0)
_sc, _pp = CommonOceanFileReader(_T0).open()
_ppx = list(_pp.planning_problem_dict.values())[0]
def fresh(): return ShieldedUSVEnv(_sc, _ppx)

print("===== A) mask 各档正确性（_as_to_mask）=====")
env = fresh(); env.reset(seed=0)
# ρ1 keep: As={a_keep} → 仅 a_keep 下标 True
m1 = env._as_to_mask({A_KEEP}, RHO_STAND_ON)
check("① ρ1 keep → 仅 a_keep 下标合法", m1.sum() == 1 and m1[_AKEEP_IDX] and not m1[IDX_EMERGENCY])
# ρ3 crossing: As=右转候选 ATR → 对应 2 个下标 True
m3 = env._as_to_mask(set(ATR), RHO_CROSSING)
exp3 = {ACTION_TO_IDX[_key(a)] for a in ATR}
check("② ρ3 crossing → ATR 两候选下标合法、idx49=False",
      set(np.where(m3)[0]) == exp3 and not m3[IDX_EMERGENCY])
# ρ5: As={a_em}（任意非网格值）→ 仅 idx49
m5 = env._as_to_mask({(0.0531, -0.0207)}, RHO_EMERGENCY)
check("③ ρ5 → 仅紧急槽 49 合法（a_em 非网格值不映射网格）", m5.sum() == 1 and m5[IDX_EMERGENCY])
# As=∅ give-way 无解 → a_keep 兜底
me = env._as_to_mask(set(), RHO_CROSSING)
check("④ As=∅（give-way 无解）→ a_keep 兜底、idx49=False",
      me.sum() == 1 and me[_AKEEP_IDX] and not me[IDX_EMERGENCY])

print("\n===== B) 49 槽不变量（step4b D15 带入）=====")
# 非 ρ5（ρ0-4 各档）idx49 必 False；ρ5 仅 idx49 True
inv = True
for rho, a_s in [(RHO_STAND_ON, {A_KEEP}), (RHO_HEAD_ON, set(ATR)), (RHO_CROSSING, set(ATR)),
                 (RHO_OVERTAKE, set(ATR)), (RHO_CROSSING, set())]:
    m = env._as_to_mask(a_s, rho)
    if m[IDX_EMERGENCY]: inv = False
check("⑤ ρ1-4 + As=∅ 各档 idx49 恒 False（49 不被误开）", inv)
m5b = env._as_to_mask({(0.05, 0.0)}, RHO_EMERGENCY)
check("⑥ ρ5 → idx49=True 且仅 1 个合法（不漏开网格动作）", m5b[IDX_EMERGENCY] and m5b.sum() == 1)
# _regular_mask（ρ0 路径）：前 49 True、idx49 False
mr = env._regular_mask()
check("⑦ ρ0 _regular_mask → 前49全True、idx49=False", mr[:49].all() and not mr[IDX_EMERGENCY] and mr.sum() == 49)

print("\n===== C) 端到端（T-0）+ 他船真实 length + 确定性 =====")
env = fresh(); obs, info = env.reset(seed=0)
check("⑧ reset → obs∈Box、含 rho/action_mask", env.observation_space.contains(obs)
      and "rho" in info and info["action_mask"].shape == (N_ACTIONS_TOTAL,))
check("⑨ 他船 length = 真实 obstacle_shape.length（≈236.4，非硬编175）", abs(env._obs_length - 236.4) < 1.0)
# action_masks() 接口
check("⑩ action_masks() = 50 bool", env.action_masks().shape == (N_ACTIONS_TOTAL,) and env.action_masks().dtype == bool)
# 跑全 episode，每步在 mask 内选动作，确认不崩 + 49 不变量全程成立
env2 = fresh(); env2.reset(seed=0)
steps = 0; inv_ok = True
while True:
    legal = np.where(env2.action_masks())[0]
    a = int(legal[0])
    obs, r, term, trunc, info = env2.step(a)
    mk = info["action_mask"]
    if mk[IDX_EMERGENCY] != (info["rho"] == RHO_EMERGENCY): inv_ok = False
    if mk[IDX_EMERGENCY] and mk.sum() != 1: inv_ok = False
    steps += 1
    if term or trunc or steps > 200: break
check("⑪ 全 episode 跑通（每步 mask 内动作不崩）", steps >= 1 and (term or trunc))
check("⑫ 全 episode 49 不变量恒成立（idx49 ⟺ ρ5 且独占）", inv_ok)
# 确定性：同 seed 两 env 同动作序列 → 逐字节一致
ea, eb = fresh(), fresh(); ea.reset(seed=1); eb.reset(seed=1)
det = True
for _ in range(8):
    a = int(np.where(ea.action_masks())[0][0])
    oa, ra, ta, ua, _ = ea.step(a)
    ob, rb, tb, ub, _ = eb.step(a)
    if not (np.array_equal(oa, ob) and ra == rb): det = False
    if ta or ua: break
check("⑬ 同 seed 同动作 → obs+reward 逐字节一致（确定性）", det)

print("\n===== D) a_em 自动传递（monkey-patch scheduler 强制 ρ5）=====")
env = fresh(); env.reset(seed=0)
A_EM = (0.0531, -0.0207)
env.scheduler.step = lambda se, so: (RHO_EMERGENCY, {A_EM})   # 强制 ρ5 + 指定 a_em
env._compute_shield()
check("⑭ 强制 ρ5 → mask 仅 idx49、_a_em 已备", env._mask[IDX_EMERGENCY] and env._mask.sum() == 1
      and env._a_em is not None and abs(env._a_em[0] - A_EM[0]) < 1e-9)
# step(49) → 底层 env 收到 emergency_action=a_em（obs 的 a_ego 反映 clip 后施加值）
obs, r, term, trunc, info = env.step(IDX_EMERGENCY)
# a_em a=0.0531 在 a_max=0.24 内不截 → obs[2]=a_ego≈0.0531
check("⑮ step(49) 用调度器 a_em（obs a_ego≈a_em[0]）", abs(obs[2] - A_EM[0]) < 1e-6)

print("\n===== E) fail-fast（49 槽不变量守护）=====")
env = fresh(); env.reset(seed=0)        # 初始 ρ0，_a_em=None
ok = False
try:
    env.step(IDX_EMERGENCY)             # 非 ρ5 选 49 → RuntimeError（mask 本应屏蔽）
except RuntimeError:
    ok = True
check("⑯ 非 ρ5 选 49（_a_em=None）→ RuntimeError（49 不变量守护）", ok)
ok2 = False
try:
    e2 = fresh(); e2.reset(seed=0); e2.step(24.9)            # 非整数下标不被 int() 截断绕过守护（Agent B MINOR）
except ValueError:
    ok2 = True
check("⑯b 非整数下标(24.9) → ValueError（不被 int 截断绕过）", ok2)
ok3 = False
try:
    fresh().action_masks()                                   # pre-reset → 友好 RuntimeError（Agent B OBS）
except RuntimeError:
    ok3 = True
check("⑯c action_masks() 前未 reset → RuntimeError", ok3)

print("\n===== F) MaskablePPO 端到端集成（train 脚手架 + torch+sb3+shield+mask）=====")
import torch
from trb_env.train import make_discrete_safe_model, POLICY_NET_ARCH
model = make_discrete_safe_model(_sc, _ppx, seed=0, n_steps=32, batch_size=16, n_epochs=1)
pi = [m.out_features for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
check("⑰ 脚手架策略 = MLP 2×64（论文 §VII）", pi == POLICY_NET_ARCH == [64, 64])
model.learn(total_timesteps=32)
check("⑱ MaskablePPO learn 端到端不崩（torch+sb3+shield+mask）", True)
# mask 真约束 agent：独立 shield env 取 obs，只留下标 7 → predict 必选 7
oenv = fresh(); o7, _ = oenv.reset(seed=0)
forced = np.zeros(N_ACTIONS_TOTAL, dtype=bool); forced[7] = True
act, _ = model.predict(o7, action_masks=forced, deterministic=True)
check("⑲ mask 真约束 agent（只留下标7 → predict=7，安全盾真生效）", int(act) == 7)

print("\n===== G) give-way ρ2-4 端到端闭环（固化审核缺口：自然 T-0 只触达 ρ0/ρ1/ρ5）=====")
# 做法（2026-06-11 审核验证过、本窗口固化）：真实 ShieldedUSVEnv，覆盖 _obs_vs() 注入【世界系恒速他船】；
# ego 走真实物理，_compute_shield / SafeActionScheduler(ColregsStatechart+EmergencyController) / _as_to_mask
# 全走【生产代码】；ρ2-4 经 ρ0→give-way persistent 真实转移进入（非 monkeypatch ρ）。
# 验证 give-way 通路：mask 恰等于 encounter_action_verification 的 As 网格下标 + 含右转 starboard + As=∅→a_keep。
from trb_env.usv_colregs import (encounter_action_verification, ATL, VesselState,
                                 crossing as _crossing, is_emergency as _is_emergency)
_ATR_IDX = {ACTION_TO_IDX[_key(a)] for a in ATR}     # 右转/starboard 网格下标 {21,22}
_ATL_IDX = {ACTION_TO_IDX[_key(a)] for a in ATL}     # 左转网格下标 {26,27}

class _CVObs:
    """世界系恒速他船：t=0 在 p0、航向 th、速度 v、长 Lo；t 时刻线性外推。"""
    def __init__(self, p0, th, v, Lo):
        self.p0 = np.asarray(p0, float); self.th = float(th); self.v = float(v); self.Lo = float(Lo)
        self.vec = self.v * np.array([np.cos(self.th), np.sin(self.th)])
    def at(self, t):
        return VesselState(position=self.p0 + self.vec * t, orientation=self.th, velocity=self.v, length=self.Lo)

def _attach(env, cv):
    dt = env.env.dt
    env._obs_vs = lambda: cv.at(env.env.step_idx * dt)   # 生产 _compute_shield 仍调它
    env._obs_length = cv.Lo

def _giveway_env(cv):
    e = fresh(); e.reset(seed=0); _attach(e, cv); e._compute_shield(); return e

_e0 = fresh(); _e0.reset(seed=0)                          # 读真实 ego 初态用于摆他船
_EX, _EY, _ETH = float(_e0.env.ego[0]), float(_e0.env.ego[1]), float(_e0.env.ego[2])
_C, _S = np.cos(_ETH), np.sin(_ETH)
def _b2w(bx, by): return np.array([_EX + bx*_C - by*_S, _EY + bx*_S + by*_C])   # ego 体坐标→世界系

def _drive_to(env, target, max_steps=12):
    for _ in range(max_steps):
        if env._rho == target: break
        env.step(int(np.where(env._mask)[0][0]))         # mask 内第一个合法动作推进物理
    return env._rho == target

def _giveway_case(label, target, psi, cv, expect):
    env = _giveway_env(cv)
    reached = _drive_to(env, target)
    got = set(np.where(env._mask)[0].tolist())
    As = encounter_action_verification(env._ego_vs(), env._obs_vs(), psi)   # 独立重算 As 对拍
    exp = {ACTION_TO_IDX[_key(a)] for a in As}
    ok = reached and got == exp and not env._mask[IDX_EMERGENCY]
    if expect == "ATR": ok = ok and got == _ATR_IDX      # head_on/crossing 永远右转
    elif expect == "ATL": ok = ok and got == _ATL_IDX
    check(label, ok)

# ⑳ ρ3 crossing（右转 ATR）/ ㉑ ρ2 head_on（右转 ATR）/ ㉒ ρ4 overtake（mask==As ⊂ 网格）
_giveway_case("⑳ ρ3 crossing 端到端 → mask==As=ATR 右转、idx49 off", RHO_CROSSING, "crossing",
              _CVObs(_b2w(3300*np.cos(np.radians(-45)), 3300*np.sin(np.radians(-45))), _ETH+np.radians(100.0), 5.0, 120.0), "ATR")
_giveway_case("㉑ ρ2 head_on 端到端 → mask==As=ATR 右转、idx49 off", RHO_HEAD_ON, "head_on",
              _CVObs(_b2w(4350.0, 0.0), _ETH+np.pi, 5.0, 120.0), "ATR")
_giveway_case("㉒ ρ4 overtake 端到端 → mask==As ⊂ 网格、idx49 off", RHO_OVERTAKE, "overtake",
              _CVObs(_b2w(1790.0, 0.0), _ETH, 1.0, 120.0), "any")

# ㉓ 真实 give-way As=∅ → a_keep 兜底（fast-crossing 他船 ov=9.5：任何右转在 t_max_m 内撞预测占据、emerg=False）
_ec = fresh(); _ec.reset(seed=0); _ec.env.ego[3] = 5.0    # 合法物理态 v=5.0
def _stand(): return VesselState(position=_b2w(1900*np.cos(np.radians(45)), 1900*np.sin(np.radians(45))),
                                 orientation=_ETH+np.radians(-70.0), velocity=5.0, length=100.0)
def _empty(): return VesselState(position=_b2w(2770*np.cos(np.radians(-65)), 2770*np.sin(np.radians(-65))),
                                 orientation=_ETH+np.radians(110.0), velocity=9.5, length=236.4)
_se = _ec._ego_vs(); _As0 = encounter_action_verification(_se, _empty(), "crossing")
_pre = _crossing(_se, _empty()) and not _is_emergency(_se, _empty()) and len(_As0) == 0   # 坐实 give-way 无解
_ec._obs_vs = _stand; _ec._obs_length = 100.0
for _ in range(4): _ec._compute_shield()                 # phase1 生产链到 ρ1
_p1 = _ec._rho == RHO_STAND_ON
_ec._obs_vs = _empty; _ec._obs_length = 236.4; _ec._compute_shield()   # phase2 ρ1→ρ3 即时支 + As=∅ 兜底
_m = _ec._mask
check("㉓ 真实 give-way As=∅ → 仅 a_keep 兜底（ρ3、不空/不崩/idx49 off/无右转误放）",
      _pre and _p1 and _ec._rho == RHO_CROSSING and _m.sum() == 1 and _m[_AKEEP_IDX]
      and not _m[IDX_EMERGENCY] and not (set(np.where(_m)[0]) & _ATR_IDX))

print("\n===== H) UnshieldedUSVEnv 无盾对照（Base/RR，step4d-②）=====")
from trb_env.usv_shield import UnshieldedUSVEnv
from trb_env.usv_colregs import RHO_NO_CONFLICT
from trb_env.evaluate import run_episode

_ue = UnshieldedUSVEnv(_sc, _ppx); _uo, _ui = _ue.reset(seed=0)
_um = _ue.action_masks()
check("㉔ 无盾 mask = 全49 True + 紧急槽 idx49 False（无 As(ρ) 约束、无盾不变量）",
      _um[:len(DISCRETE_ACTIONS)].all() and not _um[IDX_EMERGENCY] and _um.sum() == len(DISCRETE_ACTIONS))
check("㉕ reset info rho==NO_CONFLICT（无状态机 → 评估紧急步%=0、对应 Table III '–'）",
      _ui["rho"] == RHO_NO_CONFLICT)
_uo, _ur, _ut, _utr, _ui2 = _ue.step(24)                # a_keep（常规动作合法）
check("㉖ step(常规) 跑通 + rho==NO_CONFLICT", _ui2["rho"] == RHO_NO_CONFLICT)
try:
    _ue.step(IDX_EMERGENCY); _raised = False
except RuntimeError:
    _raised = True
check("㉗ step(49) 紧急槽 → RuntimeError（无盾无紧急控制器 / 机制）", _raised)
_ue2 = UnshieldedUSVEnv(_sc, _ppx); _ue2.reset(seed=0); _ev = _ue2._ego_vs()
check("㉘ _ego_vs 与底层 env.ego 一致",
      np.allclose(_ev.position, _ue2.env.ego[:2]) and _ev.orientation == _ue2.env.ego[2]
      and _ev.length == float(_ue2.env.p.l))
# ⭐ ㉙ 无盾测裸违规：注入 crossing give-way 他船（复用 G 段 _CVObs/_b2w）+ keep-straight 策略
#    （无盾 mask 全49、a_keep=24 恒合法 → ego 不给路 → ViolationCounter 记 give-way 违规 = Base 2.65 锚点机制；
#     对照 G 段：同 crossing 在 ShieldedUSVEnv 下 mask 强制 ATR、a_keep 非法 → 盾防违规。这就是盾的价值）
_ue3 = UnshieldedUSVEnv(_sc, _ppx)
_cvx = _CVObs(_b2w(3300 * np.cos(np.radians(-45)), 3300 * np.sin(np.radians(-45))),
              _ETH + np.radians(100.0), 5.0, 120.0)
_ue3._obs_vs = lambda: _cvx.at(_ue3.env.step_idx * _ue3.env.dt)   # 注入 crossing 他船（生产 _ego_vs 不变）
_ue3._obs_length = 120.0
_resu = run_episode(_ue3, lambda obs, mask: 24, max_steps=40)      # keep-straight（a_keep 恒合法）
check(f"㉙ 无盾 keep-straight 遇 crossing give-way → ViolationCounter 测得裸违规≥1（得 {_resu['violations']}，Base 锚点机制）",
      _resu["violations"] >= 1)
# ㉚ 生产 _obs_vs 同口径锁（不 monkeypatch，补 Agent 2 缺口）：_obs_vs 只依赖 step_idx + 场景他船
#    （与 ego 无关）→ 无盾 vs 有盾在同 step_idx 必逐字段一致；步进各自合法动作推进同一 step_idx。
#    锁住四方对比（Base/RR vs Safe）违规口径同源 = step4d-② 核心契约；抓生产 _obs_vs 漂移（船长/时间下标）。
_su = UnshieldedUSVEnv(_sc, _ppx); _su.reset(seed=0)
_ss = fresh(); _ss.reset(seed=0)
_okq = True; _nq = 0
for _ in range(60):
    _ovu, _ovs = _su._obs_vs(), _ss._obs_vs()            # 生产 _obs_vs（无 monkeypatch）
    if _ovu is None and _ovs is None:
        pass
    elif (_ovu is not None and _ovs is not None and np.allclose(_ovu.position, _ovs.position)
          and _ovu.orientation == _ovs.orientation and _ovu.velocity == _ovs.velocity
          and _ovu.length == _ovs.length):
        _nq += 1
    else:
        _okq = False; break
    _su.step(24)                                          # 无盾 a_keep 恒合法
    _ss.step(24 if _ss.action_masks()[24] else int(np.where(_ss.action_masks())[0][0]))  # 有盾取合法
check(f"㉚ 生产 _obs_vs 同口径锁（无盾 vs 有盾 {_nq} 步逐字段一致、不 monkeypatch）→ 四方对比口径同源、抓生产漂移",
      _okq and _nq >= 30)
# ㉛ reward 开关透传链（UnshieldedUSVEnv → USVEnv → RewardFunction，4d-②）：colregs_weight 落到 reward_fn
check("㉛ colregs_weight 透传链：Unshielded(0)→reward_fn==0 / 默认→1 / Shielded 默认→1（Base/RR/Safe 链通）",
      UnshieldedUSVEnv(_sc, _ppx, colregs_weight=0.0).env.reward_fn.colregs_weight == 0.0
      and UnshieldedUSVEnv(_sc, _ppx).env.reward_fn.colregs_weight == 1.0
      and fresh().env.reward_fn.colregs_weight == 1.0)

print("\n===== I) 多他船 runtime guard（D40#4/L49#2：盾只护 obstacles[0]，多船须 fail-fast）=====")
# 构造双他船场景：复制现有他船 + 安全 id → Shielded/Unshielded 构造均须 raise NotImplementedError（守护已接线、删 __init__ 调用则 FAIL）
import copy as _copy
_sc2 = _copy.deepcopy(_sc)
_ob1 = _copy.deepcopy(_sc2.dynamic_obstacles[0])
_ob1._obstacle_id = _sc2.generate_object_id()
_sc2.add_objects(_ob1)
def _raises_nie(cls):
    try:
        cls(_sc2, _ppx); return False
    except NotImplementedError:
        return True
check("Ⓜ ShieldedUSVEnv 双他船场景 → 构造 raise NotImplementedError（多船 fail-fast、单船 T-0 不触发）",
      _raises_nie(ShieldedUSVEnv) and len(_sc2.dynamic_obstacles) == 2)
check("Ⓜ UnshieldedUSVEnv 双他船场景 → 构造 raise NotImplementedError（Base/RR 同护、四方一致）",
      _raises_nie(UnshieldedUSVEnv))

print("\n" + ("✅ 全部 PASS" if _fail == 0 else f"❌ {_fail} 项 FAIL"))
sys.exit(1 if _fail else 0)
