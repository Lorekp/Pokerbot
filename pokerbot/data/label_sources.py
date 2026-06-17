"""Label-source string constants (torch-free; shared by encoding and training)."""

LABEL_SOURCE_SOLVER = "toolpoker_solver"
LABEL_SOURCE_POSTFLOP_SOLVER = "postflop_solver"
LABEL_SOURCES_SOLVER = frozenset({LABEL_SOURCE_SOLVER, LABEL_SOURCE_POSTFLOP_SOLVER})
