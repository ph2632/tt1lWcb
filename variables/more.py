"""Multi-jet kinematics and angular variables (per-event, computed in
build_derived/_add_more_vars).

Covers subleading AK8/AK4 jets, second/third soft-drop masses, and angular
separations between the leading AK8 jet, the subleading AK8 jet, the
lepton and the MET.
"""
import numpy as np
from .helpers import V

_PI = float(np.pi)

VARIABLES = [
    # subleading / third AK8 jets
    V("ak8_pt_1", "ak8_pt_1", r"2nd AK8 $p_T$ [GeV]", xlim=(0.0, 600.0)),
    V("ak8_pt_2", "ak8_pt_2", r"3rd AK8 $p_T$ [GeV]", xlim=(0.0, 600.0)),
    V("ak8_sdmass_1", "ak8_sdmass_1", r"2nd AK8 $m_{\mathrm{SD}}$ [GeV]",
      xlim=(0.0, 250.0)),
    V("ak8_sdmass_2", "ak8_sdmass_2", r"3rd AK8 $m_{\mathrm{SD}}$ [GeV]",
      xlim=(0.0, 250.0)),
    V("ak8_eta_1", "ak8_eta_1", r"2nd AK8 $\eta$", xlim=(-2.5, 2.5)),
    V("ak8_eta_2", "ak8_eta_2", r"3rd AK8 $\eta$", xlim=(-2.5, 2.5)),
    # subleading AK4
    V("ak4_pt_1", "ak4_pt_1", r"2nd AK4 $p_T$ [GeV]", xlim=(0.0, 400.0)),
    V("ak4_eta_1", "ak4_eta_1", r"2nd AK4 $\eta$", xlim=(-2.5, 2.5)),
    # angular separations
    V("dphi_J1_J2", "dphi_J1_J2", r"$\Delta\phi(J_1,J_2)$", xlim=(0.0, _PI)),
    V("dR_J1_J2", "dR_J1_J2", r"$\Delta R(J_1,J_2)$", xlim=(0.0, 6.0)),
    V("dphi_J1_lep", "dphi_J1_lep", r"$\Delta\phi(J_1,\ell)$",
      xlim=(0.0, _PI)),
    V("dphi_J1_met", "dphi_J1_met", r"$\Delta\phi(J_1,\mathrm{MET})$",
      xlim=(0.0, _PI)),
    V("dphi_lep_met", "dphi_lep_met", r"$\Delta\phi(\ell,\mathrm{MET})$",
      xlim=(0.0, _PI)),
    # flags
    V("has_ak8_2", "has_ak8_2", "has 2nd AK8 jet", edges=(-0.5, 0.5, 1.5)),
    V("has_ak8_3", "has_ak8_3", "has 3rd AK8 jet", edges=(-0.5, 0.5, 1.5)),
]
