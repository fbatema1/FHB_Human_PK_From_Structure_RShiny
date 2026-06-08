# ADR-007: λz and t½ Derivation Strategy

**Date:** 2026-06-08  
**Status:** Decided

## Context
The original project scope included predicting λz directly from SMILES alongside 
CL, Vd, and t½. However, λz as measured by NCA is not a true intrinsic molecular 
property — it is study-dependent (influenced by sampling window, dose, and assay 
conditions). Additionally:

- CL and Vd as reported in literature are already dose-normalized (units: mL/min/kg 
  and L/kg respectively), making them intrinsic structural properties suitable for 
  ML prediction
- λz requires knowledge of the full concentration-time profile and is not 
  consistently available across databases
- t½ and λz are mathematically linked to CL and Vd under linear kinetic assumptions

## Decision
**Do not predict λz or t½ directly.** Instead, derive both from predicted CL and Vd:

    λz = CL / Vd
    t½ = 0.693 / λz = 0.693 × Vd / CL

The model predicts **two independent intrinsic parameters** (CL and Vd) and derives 
all downstream parameters from them.

## Rationale
- CL and Vd are dose-independent intrinsic properties once normalized to body weight
- Deriving λz and t½ preserves mechanistic consistency between all reported parameters
- λz data is sparse and study-dependent — not suitable as a direct training target
- Eliminates the need to source λz values during dataset curation
- Internal consistency: predicted t½ will always equal 0.693 × Vd / CL, which 
  is not guaranteed if all three are predicted independently

## Alternatives Considered
- **Predict all four independently:** Rejected — λz is not an intrinsic structural 
  property; would require NCA study data that is inconsistently reported
- **Predict t½ directly, derive λz:** Rejected — same dose-dependency issue; 
  t½ from literature often reflects specific study conditions
- **Drop λz and t½ entirely:** Considered, but deriving them adds clinical value 
  to the tool at no modeling cost

## Consequences
- Training dataset only requires CL and Vd (plus fup for completeness)
- λz and t½ CIs must propagate uncertainty from CL and Vd predictions
- Paper must clearly state that λz and t½ are derived under assumption of 
  linear (first-order) elimination kinetics
- Shiny app output should display the derivation formula transparently
