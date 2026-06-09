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
  function applyStyle(v, style, colour, monoColour) {
    var hexColour = monoColour || "#0072B2";
    v.setStyle({}, {});   // clear existing

    var cs;
    if (colour === "spectrum") {
      // ROYGB gradient along atom serial number — works for small molecules
      cs = { colorscheme: { gradient: "ROYGB", prop: "serial", min: 0, max: 50 } };
    } else if (colour === "mono") {
      cs = { color: hexColour };
    } else {
      cs = { colorscheme: "Jmol" };
    }

    switch (style) {
      case "sphere":
        v.setStyle({}, { sphere: Object.assign({ scale: 0.4 }, cs) });
        break;
      case "line":
        v.setStyle({}, { line: cs });
        break;
      default:   // stick
        v.setStyle({}, { stick: Object.assign({ radius: 0.15 }, cs) });
    }

    // Translucent SAS surface overlay
    v.addSurface($3Dmol.SurfaceType.SAS, {
      opacity: 0.07,
      color:   hexColour
    });
  }

  // ── Color legend ────────────────────────────────────────────────────────────
  var JMOL_LEGEND = [
    { symbol: "C",  color: "#909090", label: "Carbon"   },
    { symbol: "H",  color: "#FFFFFF", label: "Hydrogen", border: "#ccc" },
    { symbol: "N",  color: "#3050F8", label: "Nitrogen"  },
    { symbol: "O",  color: "#FF0D0D", label: "Oxygen"    },
    { symbol: "S",  color: "#FFFF30", label: "Sulfur",   border: "#ccc" },
    { symbol: "F",  color: "#90E050", label: "Fluorine"  },
    { symbol: "Cl", color: "#1FF01F", label: "Chlorine"  },
    { symbol: "Br", color: "#A62929", label: "Bromine"   },
    { symbol: "P",  color: "#FF8000", label: "Phosphorus"},
    { symbol: "I",  color: "#940094", label: "Iodine"    }
  ];

  function buildLegend(colourScheme, monoColour) {
    var el = document.getElementById("mol_legend");
    if (!el) return;
    el.innerHTML = "";

    if (colourScheme === "spectrum") {
      // Gradient bar for spectrum mode
      el.innerHTML =
        '<span style="font-size:0.75rem;color:#6c757d;margin-right:6px;">Spectrum:</span>' +
        '<span style="display:inline-block;width:120px;height:12px;border-radius:6px;' +
        'background:linear-gradient(to right,#0000ff,#00ffff,#00ff00,#ffff00,#ff0000);' +
        'vertical-align:middle;border:1px solid #ccc;"></span>';
      return;
    }
    if (colourScheme === "mono") {
      var hex = monoColour || "#0072B2";
      el.innerHTML =
        '<span style="font-size:0.75rem;color:#6c757d;margin-right:6px;">Monochrome:</span>' +
        '<span style="display:inline-block;width:16px;height:16px;border-radius:3px;' +
        'background:' + hex + ';vertical-align:middle;border:1px solid #888;"></span>' +
        '<span style="font-size:0.75rem;color:#6c757d;margin-left:4px;">' + hex + '</span>';
      return;
    }

    JMOL_LEGEND.forEach(function(e) {
      var swatch = document.createElement("span");
      swatch.title = e.label;
      swatch.style.cssText = [
        "display:inline-flex", "align-items:center", "gap:3px",
        "margin:2px 4px", "font-size:0.72rem", "color:#333"
      ].join(";");

      var dot = document.createElement("span");
      dot.style.cssText = [
        "display:inline-block", "width:12px", "height:12px",
        "border-radius:50%",
        "background:" + e.color,
        "border:1px solid " + (e.border || "#888"),
        "flex-shrink:0"
      ].join(";");

      swatch.appendChild(dot);
      swatch.appendChild(document.createTextNode(e.symbol));
      el.appendChild(swatch);
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
  function loadFromSdf(sdf, style, colour, monoColour, name) {
    var v = initViewer();
    if (!v) return;

    v.clear();
    v.addModel(sdf, "sdf");
    applyStyle(v, style, colour, monoColour);
    v.zoomTo();
    v.render();

    hideSpinner();
    clearStatus();
    buildLegend(colour, monoColour);
    console.log("[molviewer] Rendered:", name);
  }

  function fetchAndRender(msg) {
    var smiles     = msg.smiles;
    var style      = msg.style      || "stick";
    var colour     = msg.colour     || "element";
    var monoColour = msg.monoColour || "#0072B2";
    var name       = msg.name       || smiles;

    showSpinner("Fetching 3D structure…");

    // ── Try PubChem first ─────────────────────────────────────────────────
    fetch(pubchemUrl(smiles))
      .then(function (res) {
        if (!res.ok) throw new Error("PubChem status " + res.status);
        return res.text();
      })
      .then(function (sdf) {
        if (!sdf || sdf.trim().length === 0) throw new Error("Empty SDF");
        loadFromSdf(sdf, style, colour, monoColour, name);
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
            loadFromSdf(sdf, style, colour, monoColour, name);
          })
          .catch(function (err2) {
            console.error("[molviewer] Both sources failed:", err2.message);
            // Show red "No Structure Available" banner
            var el = document.getElementById(VIEWER_ID);
            if (el) {
              el.innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;' +
                'height:100%;width:100%;">' +
                '<div style="background:#FDEDED;border:1.5px solid #D55E00;' +
                'border-radius:0.5rem;padding:1.2rem 2rem;text-align:center;">' +
                '<span style="font-size:1.5rem;">⚠️</span><br>' +
                '<strong style="color:#D55E00;font-size:1rem;">No Structure Available</strong><br>' +
                '<span style="color:#6c757d;font-size:0.8rem;">This compound could not be found in PubChem or NIH CACTUS.<br>' +
                'The SMILES may be novel, proprietary, or contain unsupported syntax.</span>' +
                '</div></div>';
            }
            hideSpinner();
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
    var colour     = msg.colour     || "element";
    var monoColour = msg.monoColour || "#0072B2";
    applyStyle(v, msg.style || "stick", colour, monoColour);
    v.render();
    buildLegend(colour, monoColour);
  });

})();
