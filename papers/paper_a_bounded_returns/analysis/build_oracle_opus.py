#!/usr/bin/env python
"""A5: cross-family Opus re-scoring of the oracle arm (within-version, confound-free).

Oracle cluster reports and their baselines were BOTH re-scored by the same Opus-4.8 so the
oracle-minus-baseline delta carries no judge-version artifact. Confirms whether the dual
mechanism (citation rises, factual flat) replicates on a second, cross-family judge.
Appends canonical_numbers.json['oracle']['opus_cross_check'].
"""
import json, glob, os, warnings
import numpy as np
warnings.filterwarnings("ignore")
from deep_research.paths import PROJECT_ROOT
ROOT=str(PROJECT_ROOT)
VARQ=set(json.load(open(f"{ROOT}/data/variance_stratified.json"))["query_ids"])
CLUSTER=["p1","p4","p5","p6","p7","p8"]
DIMS=["citation_quality","factual_accuracy","information_recall","coverage","analytical_depth",
      "logical_coherence","organization","instruction_following","attribution_quality"]
rng=np.random.default_rng(11)

def load(d, dim):
    """{query_id: (score, (judge_model, judge_source))} -- provenance carried so pairing enforces it.

    judge_model ALONE is not sufficient. Both arms contain 8 base rows from a separate
    'j0_version_bridge' judging campaign; in the Opus arm those carry the SAME model string
    (claude-opus-4-8) as the main campaign, so a model-only check passes them while they are
    still a different scoring run. Matching on (model, source) is what the paper's stated
    within-version guarantee actually requires.
    """
    out={}
    for f in glob.glob(f"{ROOT}/{d}/*.json"):
        q=os.path.basename(f)[:-5]
        if q in VARQ:
            j=json.load(open(f)); dd=j.get("dimensions",{})
            if dim in dd and dd[dim].get("score") is not None:
                out[q]=(float(dd[dim]["score"]), (j.get("judge_model"), j.get("judge_source")))
    return out

def paired(oracle_tmpl, base_tmpl, dim, base_fallback_tmpl=None):
    """Oracle-minus-base, enforcing that both sides carry the SAME judge version.

    The paper states the delta is "each held within its own version so the delta carries no
    version artefact", but the code only paired on query_id and never read judge_model. 8 of
    172 Sonnet base rows are scored by claude-sonnet-5 against claude-sonnet-4-6 oracle rows
    -- so the stated guarantee was asserted, not enforced. Mismatched pairs are DROPPED.

    Substitution from the sibling judging packet was tried and REJECTED: on the 172 reports
    both packets score, judge_claude_sonnet sits 0.2733 BELOW judge_claude_sonnet48 (paired SD
    0.2376). Substituting 8 base scores from the lower packet deflates those baselines and so
    inflates the delta by ~8/172 x 0.2733 = 0.0127 -- which is essentially the whole headline
    movement it produced (+0.0945 -> +0.1061). That repair would have replaced a version
    confound with a packet-scale confound and reported the artefact as a corrected result.
    Dropping loses 8 of 172 pairs and changes the estimand not at all.
    """
    deltas=[]; subs=0; dropped=0
    for p in CLUSTER:
        o=load(oracle_tmpl.format(p=p), dim)
        b=load(base_tmpl.format(p=p), dim)
        fb=load(base_fallback_tmpl.format(p=p), dim) if base_fallback_tmpl else {}
        for q,(o_score,o_ver) in o.items():
            if q not in b: continue
            b_score,b_ver = b[q]
            if b_ver != o_ver:
                # A version-matched score may exist in the sibling packet (fb), but that packet
                # is on a different scale -- see the docstring. Count it, do not use it.
                if q in fb and fb[q][1]==o_ver: subs+=1
                dropped+=1; continue
            deltas.append(o_score-b_score)
    if not deltas: return None
    a=np.array(deltas)
    boot=[rng.choice(a,len(a),replace=True).mean() for _ in range(2000)]
    return {"n":int(len(a)),"delta":round(float(a.mean()),4),
            "ci95":[round(float(np.percentile(boot,2.5)),4),round(float(np.percentile(boot,97.5)),4)],
            "n_version_matched_available_in_sibling_packet_UNUSED":subs,"n_dropped_version_mismatch":dropped}

opus={d:paired("results/judge_claude_opus/oracle_t1_{p}","results/judge_claude_opus48/base_{p}",d,
               "results/judge_claude_opus/base_{p}") for d in DIMS}
sonnet={d:paired("results/judge_claude_sonnet/oracle_t1_{p}","results/judge_claude_sonnet48/base_{p}",d,
                 "results/judge_claude_sonnet/base_{p}") for d in DIMS}
cn=json.load(open(f"{ROOT}/papers/paper_a_bounded_returns/analysis/canonical_numbers.json"))
gpt=cn["oracle"]["cluster_dims"]
def trio(dim): return {"gpt52":gpt[dim]["delta"],"opus":opus[dim]["delta"] if opus[dim] else None,
                       "sonnet":sonnet[dim]["delta"] if sonnet[dim] else None}
out={"note":"within-version Claude re-scoring (oracle and base both judged by the same Claude version); full three-judge cross-check of the GPT-5.2 dual mechanism",
     "opus_cluster_dims":opus,"sonnet_cluster_dims":sonnet,
     "panel_citation":trio("citation_quality"),"panel_factual":trio("factual_accuracy"),
     "panel_info_recall":trio("information_recall")}
cn["oracle"]["panel_cross_check"]=out
cn["oracle"]["opus_cross_check"]={"cluster_dims":opus}  # back-compat
json.dump(cn,open(f"{ROOT}/papers/paper_a_bounded_returns/analysis/canonical_numbers.json","w"),indent=1)
for dim in ["citation_quality","factual_accuracy","information_recall"]:
    t=trio(dim); print(f"{dim:20s} gpt52 {t['gpt52']:+.3f} | opus {t['opus']:+.3f} | sonnet {t['sonnet']:+.3f}")
