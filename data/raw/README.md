# Raw Data Sources

## Primary Dataset
| File | Description | Compounds | Status |
|------|-------------|-----------|--------|
| `PKIP_master_dataset_FHB_06052026_named.xlsx` | Master dataset — CL, Vd, fup with compound names | 1703 | Active |
| `master_dataset_FHB_06052026_v2.xlsx` | Alternative version of master dataset | 1666 | Superseded |
| `PKIP_expanded_dataset_FHB_06042026.xlsx` | Earlier expanded dataset | — | Superseded |

## Source Data
| File | Description |
|------|-------------|
| `CHEMBL DATA RAW.xlsx` | Raw ChEMBL PK data pull |
| `ChEMBL_new_CL_compounds.xlsx` | New CL compounds from ChEMBL |
| `CL_VD_Data.xlsx` | CL/Vd source data |
| `lombardo_raw.csv` | Lombardo dataset (raw) |
| `lombardo_with_smiles.csv` | Lombardo dataset with SMILES added |
| `lombardo_with_smiles.xlsx` | Excel version of above |
| `Enamine_FDA_approved_Drugs_plated_1123cmpds_20250601.csv` | Enamine FDA-approved compound library |

## Notes
- Primary SMILES column: `mol`
- Primary PK columns: `human_CL_mL_min_kg`, `human_VDss_L_kg`, `human_fup`
- λz and t½ columns not yet in dataset — to be added during curation or derived
- Large files (>50MB) are committed to git via Git LFS or documented here with Zenodo DOI upon publication
