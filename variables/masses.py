"""Reconstructed mass variables (per-event, computed in build_derived).

Topology (semileptonic ttbar, 1 lepton):
  - hadronic top  : t -> b W -> (b) + W(cb)  ->  AK8 jet (W->cb) + b-jet
  - leptonic top  : t -> b W -> b + ell + nu  ->  b-jet + lepton + MET
  - W (leptonic)  : lepton + MET (nu at eta=0)  ->  v_mass
  - mT(ell, nu)   : transverse mass from lepton and MET
"""
from .helpers import V

VARIABLES = [
    V("mTW", None, r"$m_{\mathrm{T}}(\ell,\nu)$ [GeV]", xlim=(0.0, 300.0)),
    V("W_mass_lepMET", None, r"$M_{\ell\nu}$ (lep+MET) [GeV]",
      xlim=(0.0, 300.0)),
    V("top_mass_fromJ1", None,
      r"$M(J_1 + b)$ (hadronic top) [GeV]", xlim=(0.0, 700.0)),
    V("top_mass_leptonic", None,
      r"$M(b + \ell + \nu)$ (leptonic top) [GeV]", xlim=(0.0, 700.0)),
]
