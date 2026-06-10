"""Injection effect on the two legible substrates: SR@0.2 vs #demos for a / e_ac / e_invdyn
(+ oracle ceiling), DINOv2 vs V-JEPA 2.1. Shows e_invdyn lifting off where plain z floors."""
import glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/tmp/selfplay_probe"
def load(sub): return [np.load(f) for f in sorted(glob.glob(f"{OUT}/sp_{sub}_results_s*.npz"))]
subs=[("dino","DINOv2-base"),("vj21","V-JEPA 2.1-base")]
conds=[("a","z only","0.6"),("c_ac","z+e_ac (fwd)","tab:orange"),("c_invdyn","z+e_invdyn","tab:red"),("oracle_xyz","oracle","green")]
fig,ax=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
for j,(sub,title) in enumerate(subs):
    R=load(sub); demos=R[0]["demos"]
    for c,lab,col in conds:
        m=np.array([np.mean([(r[f"{c}|{d}"]<0.2).mean() for r in R]) for d in demos])
        s=np.array([np.std([(r[f"{c}|{d}"]<0.2).mean() for r in R]) for d in demos])
        st="o--" if c=="oracle_xyz" else "o-"
        ax[j].plot(demos,m,st,color=col,label=lab); ax[j].fill_between(demos,m-s,m+s,color=col,alpha=.15)
    ax[j].set(xscale="log",xlabel="# HER demos",title=f"{title} ({len(R)} seeds)",ylim=(-.02,1.02))
    from matplotlib.ticker import NullLocator
    ax[j].set_xticks(list(demos)); ax[j].set_xticklabels([str(int(d)) for d in demos])
    ax[j].xaxis.set_minor_locator(NullLocator())
    ax[j].grid(alpha=.3)
ax[0].set_ylabel("success rate @ 0.2m"); ax[0].legend(fontsize=9)
fig.suptitle("Embodiment injection: inverse-dynamics e lifts low-data reaching (washes out with data)",fontsize=11)
plt.tight_layout(); plt.savefig(OUT+"/sp_inject_curve.png",dpi=120,bbox_inches="tight"); print("[ok] sp_inject_curve.png")
