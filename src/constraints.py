"""
Constraint Module Proxy
Re-exports constraint classes from data.py to maintain compatibility
with imports in integrated_framework.py
"""

from data import (
    Constraint,
    ConstraintChecker,
    TransitivityConstraint,
    SymmetryConstraint,
    AntisymmetryConstraint,
    CardinalityConstraint,
    DisjointnessConstraint,
    DomainRangeConstraint,
    Triple
)
