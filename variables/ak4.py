"""Leading AK4 jet variables (per-event, leading AK4 jet j1)."""
import numpy as np
from .helpers import V, count_edges

# AK4 tag WP codes: 5x = b-tagged (50 loose .. 54 tightest), 4x = c-tagged
# (40 .. 44), 0 = light. Non-trivial values only, in the exclusive-WP order.
AK4_TAG_EDGES = np.array([
    -0.5, 0.5, 39.5, 40.5, 41.5, 42.5, 43.5, 44.5,
    49.5, 50.5, 51.5, 52.5, 53.5, 54.5,
])

VARIABLES = [
    V("ak4_pt", "ak4_pt", r"Leading AK4 $p_{\mathrm{T}}$ [GeV]",
      xlim=(0.0, 400.0)),
    V("ak4_eta", "ak4_eta", r"Leading AK4 $\eta$", xlim=(-2.5, 2.5)),
    V("ak4_phi", "ak4_phi", r"Leading AK4 $\phi$", xlim=(-3.2, 3.2)),
    V("ak4_mass", "ak4_mass", r"Leading AK4 mass [GeV]", xlim=(0.0, 100.0)),
    V("ak4_bdisc", "ak4_bdisc", r"Leading AK4 b-discriminator",
      xlim=(-1.0, 1.0)),
    V("ak4_cvbdisc", "ak4_cvbdisc", r"Leading AK4 CvB", xlim=(-1.0, 1.0)),
    V("ak4_cvldisc", "ak4_cvldisc", r"Leading AK4 CvL", xlim=(-1.0, 1.0)),
    V("ak4_pn_b", "ak4_pn_b", r"Leading AK4 ParticleNet $b$",
      xlim=(0.0, 1.0)),
    V("ak4_pn_c", "ak4_pn_c", r"Leading AK4 ParticleNet $c$",
      xlim=(0.0, 1.0)),
    V("ak4_pn_uds", "ak4_pn_uds", r"Leading AK4 ParticleNet $uds$",
      xlim=(0.0, 1.0)),
    V("ak4_pn_g", "ak4_pn_g", r"Leading AK4 ParticleNet gluon",
      xlim=(0.0, 1.0)),
    V("ak4_tag", "ak4_tag", "Leading AK4 tag WP", edges=AK4_TAG_EDGES),
    V("ak4_hflav", "ak4_hflav", "Leading AK4 hadron flavour",
      edges=count_edges(5)),
]
