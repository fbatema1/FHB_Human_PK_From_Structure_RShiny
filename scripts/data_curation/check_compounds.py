"""
check_compounds.py
==================
Quick check of compound types and MW distribution in the master dataset.
Run: python scripts/data_curation/check_compounds.py
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_excel(
    '/Users/francisbateman/Desktop/Wadhams Start Up/Code/pk-predictor/data/raw/master_dataset_FHB_06052026_v3.xlsx',
    sheet_name='All_Compounds',
    engine='openpyxl'
)

small, large, invalid = [], [], []

for _, row in df.iterrows():
    smi  = row['mol']
    name = row['compound_name']
    if not isinstance(smi, str):
        invalid.append(name)
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        invalid.append(name)
        continue
    mw = Descriptors.MolWt(mol)
    if mw > 1000:
        large.append((name, round(mw, 1)))
    else:
        small.append((name, round(mw, 1)))

print(f"Total compounds:           {len(df)}")
print(f"Small molecule (MW<=1000): {len(small)}")
print(f"Large MW (>1000):          {len(large)}")
print(f"Invalid SMILES:            {len(invalid)}")
print()
print("Large MW compounds:")
for n, mw in sorted(large, key=lambda x: -x[1])[:20]:
    print(f"  {n:<45} MW={mw}")
print()
print("Invalid SMILES:")
for n in invalid[:20]:
    print(f"  {n}")
