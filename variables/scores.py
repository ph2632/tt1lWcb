"""Score variables: boosted Dbc (BDT & ratio) and the EventClassifier S_EVT.

These are derived per-event quantities (no dedicated ROOT branch):
  score_SC        : S_EVT = w_qq / sum(6 EC categories)
  score_Dbc       : Boosted Dbc (trained BDT)
  score_Dbc_ratio : ratio Dbc = bc/(bc+bb+bs+qcd+cc+cs+qq+topbw)
"""
from .helpers import V

VARIABLES = [
    V("score_SC", None, r"$S_{\mathrm{EVT}}$", xlim=(0.0, 1.0)),
    V("score_Dbc", None, r"$D_{bc}$ (BDT)", xlim=(0.0, 1.0)),
    V("score_Dbc_ratio", None, r"$D_{bc}$ (ratio)", xlim=(0.0, 1.0)),
    # ttH/ttZ/ttbb tagger scores (ParT-based, from the scored trees)
    V("score_ttHbb", "score_ttHbb", r"ttH$(bb)$ tagger score", xlim=(0.0, 1.0)),
    V("score_ttHcc", "score_ttHcc", r"ttH$(cc)$ tagger score", xlim=(0.0, 1.0)),
    V("score_ttLF", "score_ttLF", r"tt$+$LF tagger score", xlim=(0.0, 1.0)),
    V("score_ttZbb", "score_ttZbb", r"ttZ$(bb)$ tagger score", xlim=(0.0, 1.0)),
    V("score_ttZcc", "score_ttZcc", r"ttZ$(cc)$ tagger score", xlim=(0.0, 1.0)),
    V("score_ttZqq", "score_ttZqq", r"ttZ$(qq)$ tagger score", xlim=(0.0, 1.0)),
    V("score_ttbb", "score_ttbb", r"ttbb tagger score", xlim=(0.0, 1.0)),
    V("score_ttbj", "score_ttbj", r"ttb$+$j tagger score", xlim=(0.0, 1.0)),
    V("score_ttcc", "score_ttcc", r"ttcc tagger score", xlim=(0.0, 1.0)),
    V("score_ttcj", "score_ttcj", r"ttc$+$j tagger score", xlim=(0.0, 1.0)),
]
