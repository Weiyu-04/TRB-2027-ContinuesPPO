import sys, os, json, math
sys.path.insert(0, '.')
import warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("STEP4E_SDIR", "/tmp/trb_scenarios_pool")
import numpy as np
from shapely.geometry import Point, Polygon
import run_step4e as R
from trb_env.usv_scenarios import load_scenario_pool
from trb_env.usv_continuous_shield import ContinuousProjectionEnv
from trb_env.evaluate import evaluate_continuous
from trb_env.train import make_obs_transform
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 1) test_ids → 下载 → pool
_, test_ids = R.make_split(200, 0.3, 0, pool_size=2000)
test_paths, fails = R._download(test_ids)
print(f"[reeval] test 场景 {len(test_paths)}/{len(test_ids)} (缺{len(fails)})", flush=True)
test_pool = load_scenario_pool(test_paths)
N = len(test_pool)
CKDIR = "../结果0625-奖励改造第2次/checkpoints"

def faithful(gg, px, py, psi, ts):
    if not gg or not gg.get("vertices"): return None
    d = math.atan2(math.sin(psi-gg["orient_lo"]), math.cos(psi-gg["orient_lo"]))
    w = math.atan2(math.sin(gg["orient_hi"]-gg["orient_lo"]), math.cos(gg["orient_hi"]-gg["orient_lo"]))
    in_pos = Polygon(gg["vertices"]).intersects(Point(px,py))
    in_ori = (-1e-9 <= d <= w+1e-9)
    in_t = gg["time_lo"] <= ts <= gg["time_hi"]
    return in_pos, in_ori, in_t

allfail=[]
for s in range(5):
    base = f"{CKDIR}/Continuous-safe_s{s}_diagABwb200_s{s}"
    model = PPO.load(base+".zip", device="cpu")
    # obs_transform 重建（replay_eval 同款）
    _bv = DummyVecEnv([lambda: ContinuousProjectionEnv(*test_pool[0])])
    _vn = VecNormalize.load(base+"_vecnorm.pkl", _bv); _vn.training=False
    tf = make_obs_transform(_vn)
    agg, per = evaluate_continuous(lambda sc,pp: ContinuousProjectionEnv(sc,pp), model, test_pool,
                                   obs_transform=tf, traj_idxs=list(range(N)))
    reached=sum(1 for e in per if e["reached"]); fail=[e for e in per if not e["reached"]]
    # 自检：faithful 复现 reached == term_flags['goal']（全 60·验后处理法在真数据上零误差）
    mism=0
    for e in per:
        gg=e.get("goal_geom"); es=e.get("end_state")
        if gg and es:
            fr=faithful(gg, es["px"],es["py"],es["psi"],es["time_step"])
            if fr is not None and bool(all(fr))!=bool(e["term_flags"]["goal"]): mism+=1
    # 失败分解
    dec={"pos_miss":0,"ori_miss":0,"time_miss":0,"stopped":0,"timeout":0,"collision":0}
    for e in fail:
        tf_=e["term_flags"]; gg=e.get("goal_geom"); es=e.get("end_state")
        if tf_["stopped"]: dec["stopped"]+=1
        if tf_["time"]: dec["timeout"]+=1
        if tf_["collision"]: dec["collision"]+=1
        if gg and es:
            ip,io,it = faithful(gg, es["px"],es["py"],es["psi"],es["time_step"])
            if not ip: dec["pos_miss"]+=1
            if not io: dec["ori_miss"]+=1
            if not it: dec["time_miss"]+=1
            allfail.append({"seed":s,"px":es["px"],"py":es["py"],"psi":es["psi"],"v":es["v"],
                            "ts":es["time_step"],"in_pos":ip,"in_ori":io,"in_t":it,
                            "stopped":tf_["stopped"],"timeout":tf_["time"],"gg":gg})
    print(f"s{s}: 到达{reached}/{N} 失败{len(fail)} | faithful 自检 mism={mism} | 终止:stopped{dec['stopped']}/timeout{dec['timeout']}/coll{dec['collision']} | 失败分解 pos_miss{dec['pos_miss']}/ori_miss{dec['ori_miss']}/time_miss{dec['time_miss']}", flush=True)
    _bv.close()

# 汇总全 wb200 失败
print(f"\n=== 全 wb200 失败 n={len(allfail)} 汇总 ===")
import statistics as st
pm=sum(1 for f in allfail if not f["in_pos"]); om=sum(1 for f in allfail if not f["in_ori"]); tm=sum(1 for f in allfail if not f["in_t"])
print(f"位置门外(in_pos=False)={pm} ({100*pm/len(allfail):.0f}%) | 朝向门外(in_ori=False)={om} ({100*om/len(allfail):.0f}%) | 时间门外={tm}")
# 仅位置外/仅朝向外/都外
only_pos=sum(1 for f in allfail if not f["in_pos"] and f["in_ori"]); only_ori=sum(1 for f in allfail if f["in_pos"] and not f["in_ori"]); both=sum(1 for f in allfail if not f["in_pos"] and not f["in_ori"])
print(f"仅位置外={only_pos} | 仅朝向外={only_ori} | 位置+朝向都外={both}")
vs=[f["v"] for f in allfail]; print(f"失败终端速度 v: 中位{st.median(vs):.3f} 范围[{min(vs):.3f},{max(vs):.3f}] | v<=1e-3(失速){sum(1 for v in vs if v<=1e-3)}")
# 位置外的失败：横向(y)还是纵向(x)差？(矩形轴对齐·算到矩形的横向/纵向缺口)
print("位置外失败的终端相对矩形(样本):")
for f in allfail[:8]:
    gg=f["gg"]; vx=[v[0] for v in gg["vertices"]]; vy=[v[1] for v in gg["vertices"]]
    xmin,xmax,ymin,ymax=min(vx),max(vx),min(vy),max(vy)
    dx = 0 if xmin<=f["px"]<=xmax else min(abs(f["px"]-xmin),abs(f["px"]-xmax))
    dy = 0 if ymin<=f["py"]<=ymax else min(abs(f["py"]-ymin),abs(f["py"]-ymax))
    print(f"  s{f['seed']} end=({f['px']:.0f},{f['py']:.0f}) v={f['v']:.2f} | 纵向x缺口{dx:.0f}m 横向y缺口{dy:.0f}m | in_pos={f['in_pos']} in_ori={f['in_ori']}")
json.dump(allfail, open("/tmp/reeval_wb200_fails.json","w"))
print("\n[reeval] 完成·失败明细 → /tmp/reeval_wb200_fails.json")
