import os, math, time
os.environ.setdefault("POKERBOT_BETTING_ABSTRACTION","1"); os.environ.setdefault("POKERBOT_POSTFLOP_BUCKETS","1")
os.environ.setdefault("POKERBOT_PREFLOP_LOSSLESS","1"); os.environ.setdefault("POKERBOT_POSTFLOP_MODE","fast")
os.environ.setdefault("POKERBOT_POSTFLOP_NUM_BUCKETS","200"); os.environ.setdefault("JAX_PLATFORMS","cpu")
import pickle, numpy as np
from pokerbot.eval import lbr6
from pokerbot.eval.lbr import make_blueprint_policy
from pokerbot.eval.solver_exploitability import make_search_policy_fn
def pool(t):
    t=[x for x in t if x[2]]; N=sum(n for _,_,n in t); M=sum(m*n for m,_,n in t)/N
    ss=sum(n*(sd**2)+n*(m-M)**2 for m,sd,n in t); se=math.sqrt((ss/N)/N)
    return M*100,(M-1.96*se)*100,(M+1.96*se)*100,N
obj=pickle.load(open('checkpoints/serve_blueprints/bp6max_iter62000.pkl','rb')); snap=obj['strat_sum']
bp=make_blueprint_policy(snap); menu=(lbr6._HALF_POT_INDEX,lbr6._POT_BET_INDEX,lbr6.ALL_IN)
trip=[]; t0=time.time()
for seed in range(14):
    act=make_search_policy_fn(snap,leaf_select='uniform',seed=seed,iterations=10,search_preflop=True)
    cfg=lbr6.LBR6Config(hands=200,seed=seed,n_seats=6,track_ranges=True,paired=False,value_only_raises=True)
    cfg.bet_candidates=menu
    r=lbr6.lbr6_exploitability(act,cfg,model_fn=bp)
    trip.append((r.ev_bb_per_hand,r.stderr_bb_per_hand*(r.hands**0.5),r.hands))
    p=pool(trip); print(f"  seed {seed:2d} -> ON pool {p[0]:+.1f} [{p[1]:+.1f},{p[2]:+.1f}] ({p[3]}h, {(time.time()-t0)/60:.1f}min)",flush=True)
p=pool(trip); print(f"\nSEARCH preflop=ON (deployed) iter62000: {p[0]:+.1f} bb/100 [{p[1]:+.1f},{p[2]:+.1f}] ({p[3]}h)")
