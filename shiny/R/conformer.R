##############################################################################
# shiny/R/conformer.R
# ===================
# Generate 3D mol-blocks from SMILES using RDKit via reticulate.
#
# Called once per compound when the user opens the Structures tab.
# Results are cached in the session so repeated tab switches don't re-compute.
##############################################################################

library(reticulate)

# ── Lazy-initialise RDKit modules (once per R session) ────────────────────────
.rdkit_env <- new.env(parent = emptyenv())

init_rdkit <- function() {
  if (isTRUE(.rdkit_env$ready)) return(invisible(TRUE))

  tryCatch({
    # On shinyapps.io there is no conda — fail silently so the rest of the
    # app still works; the 3D viewer shows a friendly warning instead.
    if (!reticulate::condaenv_exists("pkip-env")) {
      .rdkit_env$ready <- FALSE
      return(invisible(FALSE))
    }
    use_condaenv("pkip-env", required = TRUE)
    .rdkit_env$Chem    <- import("rdkit.Chem",          convert = FALSE)
    .rdkit_env$AllChem <- import("rdkit.Chem.AllChem",  convert = FALSE)
    .rdkit_env$Draw    <- import("rdkit.Chem.Draw",     convert = FALSE)
    .rdkit_env$ready   <- TRUE
    message("[conformer.R] RDKit initialised via reticulate (pkip-env)")
  }, error = function(e) {
    warning("[conformer.R] Could not load RDKit: ", conditionMessage(e))
    .rdkit_env$ready <- FALSE
  })

  invisible(.rdkit_env$ready)
}

rdkit_available <- function() {
  init_rdkit()
  isTRUE(.rdkit_env$ready)
}

# ── Generate a 3D mol-block from a SMILES string ──────────────────────────────
#
# Returns a V2000 mol-block string suitable for r3dmol,
# or NULL on failure (with a warning).
#
smiles_to_molblock <- function(smiles) {
  if (!rdkit_available()) return(NULL)

  Chem    <- .rdkit_env$Chem
  AllChem <- .rdkit_env$AllChem

  tryCatch({
    mol <- Chem$MolFromSmiles(smiles)
    if (py_is_null_xptr(mol) || is.null(mol)) {
      warning("RDKit could not parse SMILES: ", smiles)
      return(NULL)
    }

    mol     <- Chem$AddHs(mol)
    ps      <- AllChem$ETKDGv3()
    result  <- AllChem$EmbedMolecule(mol, ps)

    if (py_to_r(result) == -1L) {
      # ETKDGv3 failed — try random coords fallback
      AllChem$EmbedMolecule(mol, AllChem$EmbedParameters())
    }

    AllChem$MMFFOptimizeMolecule(mol)
    mol <- Chem$RemoveHs(mol)   # keep display clean

    molblock <- py_to_r(Chem$MolToMolBlock(mol))
    molblock
  }, error = function(e) {
    warning("3D conformer generation failed for SMILES '", smiles,
            "': ", conditionMessage(e))
    NULL
  })
}

# ── Vectorised version with simple in-memory cache ────────────────────────────
.molblock_cache <- new.env(hash = TRUE, parent = emptyenv())

get_molblock <- function(smiles) {
  key <- digest::digest(smiles)   # use smiles as cache key
  if (exists(key, envir = .molblock_cache)) {
    return(get(key, envir = .molblock_cache))
  }
  mb <- smiles_to_molblock(smiles)
  assign(key, mb, envir = .molblock_cache)
  mb
}
