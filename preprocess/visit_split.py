from collections import defaultdict
import os
import shutil
import sys

import pandas as pd
from tqdm import tqdm

from preprocess.utils import extract_section

DOC_DELIM = '||||'
home_dir = os.path.expanduser('~/clin-sum/')
visits_dir = os.path.join(home_dir, 'data', 'mimic', 'visits')


if __name__ == '__main__':
    """
    MIMIC-III downloaded from https://physionet.org/content/mimiciii/1.4/#files
    """
    use_force = len(sys.argv) >= 2 and sys.argv[1] == 'force'

    if os.path.exists(visits_dir):
        if use_force:
            print('Clearing ~/clin-sum/data/mimic/visits dir')
            shutil.rmtree(visits_dir)
        else:
            raise Exception('Must clear out visits dir before running this script.  '
                            'Or run as \'python visit_split.py force\'')

    os.mkdir(visits_dir)

    print('Loading MIMIC-III notes...')
    df = pd.read_csv('~/mimic/NOTEEVENTS.csv')
    n1 = df.shape[0]
    print('{} Notes'.format(n1))
    df.dropna(subset=['HADM_ID', 'CATEGORY'], inplace=True)
    n2 = df.shape[0]
    print('{} Notes after removing documents with no visit id or assigned category'.format(n2))
    duplicate_notes = df[df.duplicated(subset=['HADM_ID', 'CATEGORY', 'DESCRIPTION'], keep=False)]
    duplicate_dsum_reports = duplicate_notes[
        (duplicate_notes['CATEGORY'] == 'Discharge summary') & (duplicate_notes['DESCRIPTION'] == 'Report')]
    flagrant_visit_ids = duplicate_dsum_reports['HADM_ID'].unique().tolist()

    print('{} visits have multiple discharge summary reports'.format(len(flagrant_visit_ids)))
    df = df[~df['HADM_ID'].isin(flagrant_visit_ids)]
    n3 = df.shape[0]
    print('Removing {} notes associated with these visits. Now have {} notes'.format(n2 - n3, n3))

    print('Loading MIMIC-III admissions...')
    visits = pd.read_csv('~/mimic/ADMISSIONS.csv')
    visits.dropna(subset=['HADM_ID'], inplace=True)
    total_visits = len(visits['HADM_ID'].unique().tolist())
    print('Total Visits={}'.format(total_visits))

    dsums_df = df[df['CATEGORY'] == 'Discharge summary']
    unique_hadm_ids = dsums_df['HADM_ID'].unique().tolist()

    dsum_visits = len(unique_hadm_ids)
    print('Num Visits w/ Dsum={}'.format(dsum_visits))

    metadata_df = defaultdict(list)
    for i in tqdm(range(dsum_visits)):
        hadm_id = unique_hadm_ids[i]
        hadm_id_str = str(int(hadm_id))
        output_dir = os.path.join(visits_dir, hadm_id_str)
        os.mkdir(output_dir)
        visit_notes = df[df['HADM_ID'] == hadm_id]
        n = visit_notes.shape[0]
        if n < 2:
            continue
        metadata_df['HADM_ID'].append(hadm_id_str)
        admission_data = visits[visits['HADM_ID'] == hadm_id]
        has_admission_metadata = admission_data.shape[0] > 0
        if has_admission_metadata:
            admission_data.to_csv(os.path.join(output_dir, 'visit_data.csv'), index=False)
        metadata_df['has_admission_metadata'] = has_admission_metadata

        dsum = visit_notes[visit_notes['CATEGORY'] == 'Discharge summary']
        other_notes = visit_notes[~(visit_notes['CATEGORY'] == 'Discharge summary')]

        metadata_df['dsum_ct'].append(dsum.shape[0])
        metadata_df['other_ct'].append(other_notes.shape[0])

        dsum.to_csv(os.path.join(output_dir, 'dsum.csv'), index=False)
        other_notes.to_csv(os.path.join(output_dir, 'input_notes.csv'), index=False)

        pmhs, hpis, hcs = [], [], []
        for i, row in dsum.iterrows():
            row = row.to_dict()
            hpi = extract_section(row['TEXT'], matches=['history of (the )?present illness', 'hpi'], partial_match=True)
            if hpi is not None and hpi not in hpis:
                hpis.append(hpi)
            pmh = extract_section(row['TEXT'], matches=['past medical history', 'pmh'], partial_match=False)
            if pmh is not None and pmh not in pmhs:
                pmhs.append(pmh)
            hc = extract_section(row['TEXT'], matches=['hospital course'], partial_match=True)
            if hc is not None and hc not in hcs:
                hcs.append(hc)

        metadata_df['pmh_ct'] = len(pmhs)
        metadata_df['hpi_ct'] = len(hpis)
        metadata_df['hcs_ct'] = len(hcs)

        with open(os.path.join(output_dir, 'hpis.txt'), 'w') as fd:
            fd.write(DOC_DELIM.join(hpis))
        with open(os.path.join(output_dir, 'pmhs.txt'), 'w') as fd:
            fd.write(DOC_DELIM.join(pmhs))
        with open(os.path.join(output_dir, 'hcs.txt'), 'w') as fd:
            fd.write(DOC_DELIM.join(hcs))

    metadata_df = pd.DataFrame(metadata_df)
    metadata_df.to_csv(os.path.join(output_dir, 'visit_metadata.csv'), index=False)
