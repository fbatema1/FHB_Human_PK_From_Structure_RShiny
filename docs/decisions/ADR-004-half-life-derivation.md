# ADR-004: Half-Life Calculation Method

**Date:** 2026-06-07  
**Status:** Decided

## Context
t½ is a commonly reported PK parameter. It can be predicted directly from SMILES 
or derived from other predicted parameters. Both approaches have trade-offs.

## Decision
**Primary output:** Derive t½ from predicted CL and Vd:

    t½ = 0.693 × Vd / CL

**Consistency check:** Also train a direct t½ predictor. Report GMFE for both 
approaches. Use whichever performs better; document in the Shiny app which method 
was used.

## Alternatives Considered
- **Direct prediction only:** Simpler, but loses the mechanistic relationship 
  between t½, CL, and Vd. Internal consistency not guaranteed.
- **Derived only:** Cleaner scientifically, but error propagation from CL and 
  Vd predictions can inflate t½ error.

## Rationale
Deriving t½ from CL and Vd preserves the pharmacokinetic relationship between 
parameters (important for a tool used in drug development). The direct predictor 
serves as a quality check — if direct prediction substantially outperforms the 
derived value, it signals that CL or Vd predictions have systematic error.

## Consequences
- t½ CI must propagate uncertainty from both CL and Vd predictions
- Shiny app must clearly indicate whether t½ is derived or directly predicted
- Both approaches must be evaluated and compared in the methods/results
