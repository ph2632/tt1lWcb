"""Leading AK8 jet variables (per-event, leading jet J1)."""
import numpy as np
from .helpers import V, count_edges

VARIABLES = [
    V("ak8_eta", "ak8_eta", r"Leading AK8 $\eta$", xlim=(-2.5, 2.5)),
    V("ak8_phi", "ak8_phi", r"Leading AK8 $\phi$", xlim=(-3.2, 3.2)),
    V("ak8_tau32", "ak8_tau32", r"Leading AK8 $\tau_{32}$", xlim=(0.0, 1.5)),
    V("ak8_nConstituents", "ak8_nConstituents",
      "Leading AK8 number of constituents", xlim=(0.0, 150.0)),
    V("ak8_n_in_jet", "ak8_n_in_jet",
      r"Leading AK8 $n_{\mathrm{quarks}}$ in jet", edges=count_edges(5)),
    V("ak8_n_b_in_jet", "ak8_n_b_in_jet",
      r"Leading AK8 $n_{b}$ in jet", edges=count_edges(5)),
    V("ak8_n_c_in_jet", "ak8_n_c_in_jet",
      r"Leading AK8 $n_{c}$ in jet", edges=count_edges(5)),
    V("ak8_type", "ak8_type", "Leading AK8 gen-level type",
      edges=count_edges(5)),
    V("ak8_is_wbc", "ak8_is_wbc", "Leading AK8 is W->cb (truth)",
      edges=count_edges(1)),
    # GloParT score components (leading jet)
    V("ak8_gpt_bc", "ak8_gpt_bc", r"GloParT $bc$ score", xlim=(0.0, 1.0)),
    V("ak8_gpt_bb", "ak8_gpt_bb", r"GloParT $bb$ score", xlim=(0.0, 1.0)),
    V("ak8_gpt_cc", "ak8_gpt_cc", r"GloParT $cc$ score", xlim=(0.0, 1.0)),
    V("ak8_gpt_qcd", "ak8_gpt_qcd", r"GloParT QCD score", xlim=(0.0, 1.0)),
    V("ak8_gpt_bs", "ak8_gpt_bs", r"GloParT $bs$ score", xlim=(0.0, 1.0)),
    V("ak8_gpt_qq", "ak8_gpt_qq", r"GloParT $qq$ score", xlim=(0.0, 1.0)),
    V("ak8_gpt_cs", "ak8_gpt_cs", r"GloParT $cs$ score", xlim=(0.0, 1.0)),
    V("ak8_gpt_topbw", "ak8_gpt_topbw", r"GloParT top($bW$) score",
      xlim=(0.0, 1.0)),
    V("ak8_gpt_bqq", "ak8_gpt_bqq", r"GloParT top($bqq'$) score",
      xlim=(0.0, 1.0)),
    V("ak8_gpt_topw", "ak8_gpt_topw", r"GloParT top($W$) score",
      xlim=(0.0, 1.0)),
]
