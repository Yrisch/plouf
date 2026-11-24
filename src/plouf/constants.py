# plouf: camera paths through SPH simulations, rendered with shamrock
# Copyright (c) 2026 the plouf authors
# SPDX-License-Identifier: CECILL-2.1
# plouf is licensed under the CeCILL 2.1 License, see LICENSE for more information

"""Physical constants and conventions shared by the modules (cgs unless stated)."""

MSUN_G = 1.9891e33  # solar mass [g]
PC_CM = 3.086e18  # parsec [cm]
MYR_S = 3.15576e13  # megayear [s]
AU_PER_PC = 206264.8

STAR_MAX_RACC_PC = 0.01  # sink mode "cluster": sinks with a smaller accretion radius [pc] are stars, the others cores

# how sinks are drawn (look["sinks"]; data/sinktracks.split_sinks): mode -> UI name
SINK_MODES = {
    "markers": "plain markers",  # every sink a disc of its accretion radius (at least a few px), one colour
    "cluster": "cluster: stars and cores",  # accretion radius < star_max_racc_pc: star (photometry), else core disc
    "stars": "all stars",  # every sink a star, photometry from its mass
    "hidden": "hidden",  # not drawn (still selectable and followable)
}
DEFAULT_SINK_MODE = "markers"
