import pandas as pd
import requests
import time

MASTER_PATH = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/PKIP_master_dataset_FHB_06052026.xlsx"
OUT_PATH    = r"/Users/francisbateman/Desktop/Wadhams Start Up/Code/PKIP_master_dataset_FHB_06052026_named.xlsx"

xl = pd.ExcelFile(MASTER_PATH, engine='calamine')
df = pd.read_excel(xl, sheet_name='Enamine_New_Pending_PK')
print(f"Looking up names for {len(df)} compounds...")

names = []
for i, row in df.iterrows():
    smi = str(row['mol'])
    name = None
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smi)}/property/Title/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            name = r.json()['PropertyTable']['Properties'][0].get('Title')
    except:
        pass
    names.append(name)
    
    if (i+1) % 50 == 0:
        found = sum(1 for n in names if n)
        print(f"  [{i+1}/{len(df)}] names found: {found}")
    time.sleep(0.2)

df['compound_name'] = names
found = df['compound_name'].notna().sum()
print(f"\nNames found: {found}/{len(df)}")

# Write back all sheets, updating Enamine_New_Pending_PK
with pd.ExcelWriter(OUT_PATH, engine='openpyxl') as writer:
    for sheet in xl.sheet_names:
        if sheet == 'Enamine_New_Pending_PK':
            df.to_excel(writer, sheet_name=sheet, index=False)
        else:
            pd.read_excel(xl, sheet_name=sheet).to_excel(writer, sheet_name=sheet, index=False)

print(f"Saved → {OUT_PATH}")