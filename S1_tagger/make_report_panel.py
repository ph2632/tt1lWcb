#!/usr/bin/env python3
"""
Render every S1 / S1' result figure from a finished run and assemble the
per-tagger 2x2 report panels.

`build_report()` is called automatically at the end of every `train.py` run;
run this script standalone to re-render from an existing summary.json + the
per-model eval.npz files (no retraining).

Per run, into  S1_tagger/output/<dataset_tag>/ :
  roc.png                 S1 / S1' / old Dbc / 3-class Dbc, same test jets
  overtrain_{S1,S1p}.png  train-vs-test score shapes + KS test  (held-out 15%)
  featimp_{S1,S1p}.png    top-15 nodes by gain + a training-info box
  score_{S1,S1p}.png      score split by W->cb topology
  results_table.png       AUC / working points / params / overtraining / top-10
  panel_{S1,S1p}.png      2x2:  ROC | overtraining / feature-importance | score

and, copied into the tt1lWcb gallery root (:8899), tagged per run:
  S1_tagger_<tag>_panel_{S1,S1p}.png, _results_table.png, _roc.png
  + S1_tagger_panel_latest_{S1,S1p}.png   (newest run, untagged)

Usage:
  ./.venv/bin/python S1_tagger/make_report_panel.py [--config S1_tagger/config.json]
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
GALLERY = HERE.parent
BLUE = "#3b6ea5"
NAMES = {"S1": r"$S_{1}$", "S1p": r"$S_{1}^{\prime}$"}


# --------------------------------------------------------------------------- #
def _load_eval(outdir, name):
    return dict(np.load(outdir / name / "eval.npz"))


def roc_figure(outdir, summary, out_png):
    curves = []
    for name in ("S1", "S1p"):
        e = _load_eval(outdir, name)
        curves.append((NAMES[name], e["test_y"], e["test_score"], e["test_w"]))
    for key, lab in (("Dbc_old_8node", r"$D_{bc}$ old (8 nodes)"),
                     ("Dbc_3class_expanded", r"$D_{bc}$ 3-class (33 nodes)")):
        f = outdir / f"cmp_{key}.npz"
        if f.exists():
            d = np.load(f)
            curves.append((lab, d["y"], d["score"], d["w"]))

    plt.figure(figsize=(6.6, 5.6))
    for lab, y, s, w in curves:
        fpr, tpr, _ = roc_curve(y, s, sample_weight=w)
        auc = roc_auc_score(y, s, sample_weight=w)
        plt.plot(tpr, fpr, lw=1.9, label=f"{lab}   AUC {auc:.4f}")
    plt.yscale("log"); plt.ylim(1e-4, 1.0)
    plt.xlabel("signal efficiency"); plt.ylabel("background efficiency")
    plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=8.5, loc="lower right")
    plt.title(f"boosted-cb tagger ROC   (test jets, PRESEL)   "
              f"[{summary['config']['dataset_tag']}]")
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def overtrain_figure(outdir, name, summary, out_png):
    e = _load_eval(outdir, name)
    ot = summary["models"][name]["overtraining"]
    bins = np.linspace(0, 1, 41)

    plt.figure(figsize=(6.6, 5.2))
    specs = [("train_score", "train_y", "train_w", 1, "signal", "#c44",  "step",    None),
             ("test_score",  "test_y",  "test_w",  1, "signal (test)", "#c44", "points", "o"),
             ("train_score", "train_y", "train_w", 0, "background", "#468", "stepfill", None),
             ("test_score",  "test_y",  "test_w",  0, "background (test)", "#468", "points", "s")]
    for sk, yk, wk, cls, lab, col, kind, mk in specs:
        m = e[yk] == cls
        s, w = e[sk][m], e[wk][m]
        h, edges = np.histogram(s, bins=bins, weights=w, density=True)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        if kind == "step":
            plt.step(ctr, h, where="mid", color=col, lw=1.6, label=lab)
        elif kind == "stepfill":
            plt.fill_between(ctr, h, step="mid", color=col, alpha=0.25, label=lab)
            plt.step(ctr, h, where="mid", color=col, lw=1.2)
        else:
            err = np.sqrt(np.histogram(s, bins=bins, weights=w**2)[0]) / max(
                (w.sum() * np.diff(edges)[0]), 1e-12)
            plt.errorbar(ctr, h, yerr=err, fmt=mk, ms=3.5, color=col,
                         lw=0, elinewidth=0.8, label=lab)
    plt.yscale("log")
    plt.xlabel(NAMES[name] + "  (BDT fine-tuned score)")
    plt.ylabel("a.u.  (density)")
    col = {"OK": "#2a8a2a", "WARN": "#c98a00", "FAIL": "#c0392b"}[ot["verdict"]]
    txt = (f"KS $p$  sig {ot['ks_sig_p']:.3f}   bkg {ot['ks_bkg_p']:.3f}\n"
           f"sig-eff bias  {ot['sig_eff_bias_pct']:+.1f}%\n"
           f"train AUC {summary['models'][name]['train']['auc']:.4f}   "
           f"test AUC {summary['models'][name]['test']['auc']:.4f}")
    plt.gca().text(0.03, 0.97, txt, transform=plt.gca().transAxes, va="top",
                   fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec=col, lw=1.6))
    plt.legend(fontsize=8, loc="lower center", ncol=2)
    plt.title(f"{NAMES[name]}  overtraining test   "
              f"(held-out 15%)   [{ot['verdict']}]")
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def featimp_figure(outdir, name, summary, out_png):
    m = summary["models"][name]
    top = m["feature_importance"][:15]
    labels = [r["feature"].replace("ak8_gpt_", "") for r in top][::-1]
    gains = [r["gain"] for r in top][::-1]

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.barh(range(len(labels)), gains, color=BLUE)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("XGBoost importance  (total gain)")
    ax.set_title(f"{NAMES[name]}  feature importance  "
                 f"(top 15 of {m['n_features']} inputs)")
    ax.grid(alpha=0.3, axis="x")

    ot = m["overtraining"]
    xgb = summary["config"]["xgboost"]
    info = (
        f"$\\bf{{training}}$\n"
        f"train jets   {m['n_train']:,}\n"
        f"  signal     {m['n_train_signal']:,}\n"
        f"  background  {m['n_train_background']:,}\n"
        f"scale_pos_w  {m['scale_pos_weight']:.0f}\n"
        f"trees        {xgb['n_estimators']} (stop @ {m['best_iteration']})\n"
        f"depth {xgb['max_depth']}   lr {xgb['learning_rate']}\n"
        f"subsample {xgb['subsample']}   colsample {xgb['colsample_bytree']}\n"
        f"fit time     {m['fit_seconds']:.0f} s\n"
        f"\n$\\bf{{result}}$\n"
        f"AUC  train {m['train']['auc']:.4f}\n"
        f"     valid {m['valid']['auc']:.4f}\n"
        f"     test  {m['test']['auc']:.4f}\n"
        f"KS $p$  sig {ot['ks_sig_p']:.3f}  bkg {ot['ks_bkg_p']:.3f}\n"
        f"overtraining  {ot['verdict']}  ({ot['sig_eff_bias_pct']:+.1f}%)")
    ax.text(0.99, 0.02, info, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#bbbbbb"))
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def score_figure(outdir, name, summary, sig_codes, out_png):
    e = _load_eval(outdir, name)
    s, w, topo = e["test_score"], e["test_w"], e["test_topo"]
    bins = np.linspace(0, 1, 51)
    plt.figure(figsize=(6.6, 4.8))
    plt.hist(s[topo == 0], bins=bins, weights=w[topo == 0], density=True,
             histtype="step", lw=1.9, color="#468", label="background")
    plt.hist(s[topo > 0], bins=bins, weights=w[topo > 0], density=True,
             histtype="step", lw=1.9, color="#c44", label="signal (all W->cb)")
    for c, cname in sig_codes.items():
        mm = topo == int(c)
        if mm.sum() > 20:
            plt.hist(s[mm], bins=bins, weights=w[mm], density=True,
                     histtype="step", lw=1.0, ls="--", label=cname)
    plt.yscale("log")
    plt.xlabel(NAMES[name] + "  (BDT fine-tuned score)"); plt.ylabel("a.u.")
    plt.legend(fontsize=8)
    plt.title(f"{NAMES[name]} score by topology   (test jets)")
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


# --------------------------------------------------------------------------- #
def _mk_table(ax, title, header, rows, col_widths=None, shade=(), align="center"):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=8)
    t = ax.table(cellText=[header] + rows, cellLoc=align, loc="upper left",
                 colWidths=col_widths, bbox=[0.0, 0.0, 1.0, 0.86])
    t.auto_set_font_size(False); t.set_fontsize(9.5)
    for j in range(len(header)):
        t[0, j].set_facecolor(BLUE)
        t[0, j].set_text_props(color="white", fontweight="bold", fontsize=8.7)
    for i in shade:
        for j in range(len(header)):
            t[i, j].set_facecolor("#e8f0f7")
    for _, c in t.get_celld().items():
        c.set_edgecolor("#c8c8c8")
    return t


def table_figure(summary, out_png):
    s1, s1p = summary["models"]["S1"], summary["models"]["S1p"]
    cmp = summary["comparison_test_auc"]
    xgb = summary["config"]["xgboost"]
    tag = summary["config"]["dataset_tag"]

    fig = plt.figure(figsize=(11.0, 8.8))
    fig.suptitle(r"$S_{1}$ / $S_{1}^{\prime}$  boosted-cb tagger  $-$  "
                 f"test jets, PRESEL   [{tag}]   {summary.get('run_stamp','')}",
                 fontsize=13, fontweight="bold", y=0.985)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.35, 2.15],
                          hspace=0.55, wspace=0.16, left=0.05, right=0.985,
                          top=0.90, bottom=0.045)

    # (a) discrimination + overtraining
    hdr = ["tagger", "inputs", "test AUC", "sig-eff @ bkg 5%", "@ bkg 1%",
           "@ bkg 0.1%", "overtrain"]
    rows = []
    for key, m in (("S1'", s1p), ("S1", s1)):
        wp = m["test"]["working_points"]
        ot = m["overtraining"]
        rows.append([key, "30+tau21,32" if key == "S1'" else "30 nodes",
                     f"{m['test']['auc']:.4f}",
                     *[f"{w['signal_eff']:.3f}" for w in wp],
                     f"{ot['verdict']} ({ot['sig_eff_bias_pct']:+.0f}%)"])
    rows.append([r"$D_{bc}$ old", "8 nodes (pkl)",
                 f"{cmp.get('Dbc_old_8node', float('nan')):.4f}", "-", "-", "-", "-"])
    rows.append([r"$D_{bc}$ 3-class", "33 nodes, [:,0]",
                 f"{cmp.get('Dbc_3class_expanded', float('nan')):.4f}", "-", "-", "-", "-"])
    _mk_table(fig.add_subplot(gs[0, :]), "(a)  discrimination + overtraining  (weighted)",
              hdr, rows, col_widths=[0.13, 0.15, 0.11, 0.15, 0.10, 0.11, 0.15],
              shade=(1, 2))

    # (b) per-class signal efficiency @ 1e-3
    pc1 = s1["test"]["per_class_signal_eff_at_1e-3"]
    pc1p = s1p["test"]["per_class_signal_eff_at_1e-3"]
    rows = [[k, f"{pc1[k]:.3f}", f"{pc1p[k]:.3f}"] for k in pc1]
    _mk_table(fig.add_subplot(gs[1, 0]), "(b)  per-class sig-eff @ bkg-eff 0.1%",
              ["W->cb topology", r"$S_{1}$", r"$S_{1}^{\prime}$"], rows,
              col_widths=[0.34, 0.16, 0.16])

    # (c) training set + params
    dc = summary["dataset_counts"]
    n_sig = sum(sum(dc[sp][c] for c in dc[sp] if c != "0") for sp in dc)
    n_bkg = sum(dc[sp]["0"] for sp in dc)
    kp = summary["config"]["background_keep_prob"]
    rows = [["signal jets (PRESEL)", f"{n_sig:,}"],
            ["background jets", f"{n_bkg:,}  (keep-prob {kp})"],
            ["split", "70 train / 15 valid / 15 test"],
            ["trees / best_iter", f"{xgb['n_estimators']} / "
             f"{s1['best_iteration']},{s1p['best_iteration']}"],
            ["depth / lrate", f"{xgb['max_depth']} / {xgb['learning_rate']}"],
            ["subsample / colsample", f"{xgb['subsample']} / {xgb['colsample_bytree']}"],
            ["scale_pos_weight", f"{s1['scale_pos_weight']:.0f}"],
            ["tree_method / obj.", f"{xgb['tree_method']} / {xgb['objective']}"]]
    _mk_table(fig.add_subplot(gs[1, 1]), "(c)  training set + XGBoost parameters",
              ["item", "value"], rows, col_widths=[0.26, 0.40], align="left")

    # (d) top-10 features by gain
    rk = s1["feature_importance"][:10]
    rows = [[str(i), r["feature"].replace("ak8_gpt_", ""),
             f"{r['gain']:.0f}", f"{r['weight']:.0f}", f"{r['cover']:.0f}"]
            for i, r in enumerate(rk, 1)]
    ax = fig.add_subplot(gs[2, :])
    _mk_table(ax, r"(d)  $S_{1}$ feature importance  $-$  top 10 of 30 GloParT nodes",
              ["#", "GloParT node", "gain", "n splits", "cover"], rows,
              col_widths=[0.08, 0.30, 0.16, 0.16, 0.16], shade=(1, 2))
    ax.text(0.0, -0.05, r"$bc$ is only #6;  $bb$ dominates $-$ it is the discriminant "
            r"for the $t^{2}(b'b)$ class, for which GloParT has no dedicated node.",
            transform=ax.transAxes, fontsize=8.5, style="italic", color="#444444")

    fig.savefig(out_png, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
def build_report(cfg, outdir):
    outdir = Path(outdir)
    summary = json.loads((outdir / "summary.json").read_text())
    tag = cfg["dataset_tag"]
    ds_meta = json.loads((Path(summary["dataset_dir"]) / "dataset_complete.json").read_text())
    sig_codes = {int(k): v for k, v in ds_meta["signal_codes"].items()}

    roc = outdir / "roc.png"
    roc_figure(outdir, summary, roc)
    table = outdir / "results_table.png"
    table_figure(summary, table)

    published = {f"S1_tagger_{tag}_results_table.png": table,
                 f"S1_tagger_{tag}_roc.png": roc}
    for name in ("S1", "S1p"):
        ov = outdir / f"overtrain_{name}.png"
        fi = outdir / f"featimp_{name}.png"
        sc = outdir / f"score_{name}.png"
        overtrain_figure(outdir, name, summary, ov)
        featimp_figure(outdir, name, summary, fi)
        score_figure(outdir, name, summary, sig_codes, sc)
        panel = outdir / f"panel_{name}.png"
        subprocess.run(
            ["montage", str(roc), str(ov), str(fi), str(sc),
             "-tile", "2x2", "-geometry", "1100x870+8+8",
             "-background", "white", "-gravity", "center", str(panel)], check=True)
        published[f"S1_tagger_{tag}_panel_{name}.png"] = panel
        shutil.copyfile(panel, GALLERY / f"S1_tagger_panel_latest_{name}.png")

    for gname, srcfile in published.items():
        shutil.copyfile(srcfile, GALLERY / gname)

    print("  report:")
    for name in ("S1", "S1p"):
        p = outdir / f"panel_{name}.png"
        print(f"    {p}  ({p.stat().st_size // 1024} KB)")
    print(f"    published -> {GALLERY}/S1_tagger_{tag}_*.png  "
          f"(+ S1_tagger_panel_latest_{{S1,S1p}}.png)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    build_report(cfg, HERE / cfg["output_dir"] / cfg["dataset_tag"])


if __name__ == "__main__":
    main()
