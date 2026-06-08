/* =============================================================================
   www/molviewer.js
   ================
   Browser-side 3D molecular viewer for Wadhams PK Predictor.

   Uses 3Dmol.js (loaded via CDN) to render interactive 3D structures.
   3D coordinates fetched from PubChem REST API, with NIH CACTUS as fallback.

   Works on desktop Chrome/Firefox/Safari and mobile browsers.
   No Python, no R packages, no conda required.

   Shiny R server sends a custom message:
     session$sendCustomMessage("loadMolecule", list(
       smiles = "CC(C)Cc1ccc...",
       name   = "Ibuprofen",
       style  = "stick",        // stick | sphere | line | cartoon
       colour = "element"       // element | chain | residue
     ))
   ============================================================================= */

(function () {
  "use strict";

  // ── Config ──────────────────────────────────────────────────────────────────
  var VIEWER_ID    = "mol_3d_viewer";
  var SPINNER_ID   = "mol_3d_spinner";
  var STATUS_ID    = "mol_3d_status";
  var viewer       = null;

  // PubChem 3D SDF endpoint (URL-encoded SMILES)
  function pubchemUrl(smiles) {
    return "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
      + encodeURIComponent(smiles)
      + "/SDF?record_type=3d";
  }

  // NIH CACTUS fallback (generates 3D coords on-the-fly from SMILES)
  function cactusUrl(smiles) {
    return "https://cactus.nci.nih.gov/chemical/structure/"
      + encodeURIComponent(smiles)
      + "/file?format=sdf&get3d=true";
  }

  // ── Initialise viewer div ───────────────────────────────────────────────────
  function initViewer() {
    var el = document.getElementById(VIEWER_ID);
    if (!el) return null;
    if (viewer) { viewer.clear(); return viewer; }

    viewer = $3Dmol.createViewer(el, {
      backgroundColor: "white",
      antialias:       true,
      id:              VIEWER_ID
    });
    return viewer;
  }

  // ── Style helpers ───────────────────────────────────────────────────────────
  var COLOUR_SCHEMES = {
    element: { colorscheme: "Jmol"         },
    chain:   { colorscheme: "chainHetatm"  },
    residue: { colorscheme: "amino"        }
  };

  function applyStyle(v, style, colour) {
    var cs = COLOUR_SCHEMES[colour] || COLOUR_SCHEMES.element;
    v.setStyle({}, {});   // clear

    switch (style) {
      case "sphere":
        v.setStyle({}, { sphere: Object.assign({ scale: 0.4 }, cs) });
        break;
      case "line":
        v.setStyle({}, { line: cs });
        break;
      case "cartoon":
        v.setStyle({}, { cartoon: { color: "spectrum" } });
        break;
      default:   // stick
        v.setStyle({}, { stick: Object.assign({ radius: 0.15 }, cs) });
    }

    // Translucent surface overlay
    v.addSurface($3Dmol.SurfaceType.SAS, {
      opacity:      0.07,
      color:        "#0072B2"
    });
  }

  // ── Status / spinner helpers ────────────────────────────────────────────────
  function showSpinner(msg) {
    var sp = document.getElementById(SPINNER_ID);
    var st = document.getElementById(STATUS_ID);
    if (sp) sp.style.display = "flex";
    if (st) st.textContent   = msg || "Loading…";
  }

  function hideSpinner() {
    var sp = document.getElementById(SPINNER_ID);
    if (sp) sp.style.display = "none";
  }

  function showError(msg) {
    hideSpinner();
    var st = document.getElementById(STATUS_ID);
    if (st) {
      st.textContent  = msg;
      st.style.color  = "#D55E00";
    }
  }

  function clearStatus() {
    var st = document.getElementById(STATUS_ID);
    if (st) { st.textContent = ""; st.style.color = "#6c757d"; }
  }

  // ── Fetch SDF and render ────────────────────────────────────────────────────
  function loadFromSdf(sdf, style, colour, name) {
    var v = initViewer();
    if (!v) return;

    v.clear();
    v.addModel(sdf, "sdf");
    applyStyle(v, style, colour);
    v.zoomTo();
    v.spin("y", 0.5);
    v.render();

    hideSpinner();
    clearStatus();
    console.log("[molviewer] Rendered:", name);
  }

  function fetchAndRender(msg) {
    var smiles = msg.smiles;
    var style  = msg.style  || "stick";
    var colour = msg.colour || "element";
    var name   = msg.name   || smiles;

    showSpinner("Fetching 3D structure…");

    // ── Try PubChem first ─────────────────────────────────────────────────
    fetch(pubchemUrl(smiles))
      .then(function (res) {
        if (!res.ok) throw new Error("PubChem status " + res.status);
        return res.text();
      })
      .then(function (sdf) {
        if (!sdf || sdf.trim().length === 0) throw new Error("Empty SDF");
        loadFromSdf(sdf, style, colour, name);
      })
      .catch(function (err) {
        console.warn("[molviewer] PubChem failed:", err.message,
                     "— trying CACTUS…");
        showSpinner("PubChem unavailable — trying CACTUS…");

        // ── Fallback: NIH CACTUS ────────────────────────────────────────
        fetch(cactusUrl(smiles))
          .then(function (res) {
            if (!res.ok) throw new Error("CACTUS status " + res.status);
            return res.text();
          })
          .then(function (sdf) {
            if (!sdf || sdf.includes("Page not found") || sdf.trim().length === 0) {
              throw new Error("CACTUS returned no structure");
            }
            loadFromSdf(sdf, style, colour, name);
          })
          .catch(function (err2) {
            console.error("[molviewer] Both sources failed:", err2.message);
            showError("Could not retrieve 3D structure. " +
                      "The compound may not be in PubChem or CACTUS.");
          });
      });
  }

  // ── Shiny message handler ───────────────────────────────────────────────────
  Shiny.addCustomMessageHandler("loadMolecule", function (msg) {
    if (!msg.smiles || msg.smiles.trim() === "") {
      showError("No SMILES provided.");
      return;
    }
    fetchAndRender(msg);
  });

  // ── Style/colour change without re-fetching ─────────────────────────────────
  // (server sends this when only display options change)
  Shiny.addCustomMessageHandler("restyleMolecule", function (msg) {
    var v = viewer;
    if (!v) return;
    applyStyle(v, msg.style || "stick", msg.colour || "element");
    v.render();
  });

})();
