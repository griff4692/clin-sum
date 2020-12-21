from collections import defaultdict
from datetime import datetime
import os
from time import time

import pandas as pd
from scipy.stats import describe

from constants import *
from utils import *


def _strip(str):
    return str.strip(' "\n')


def is_none(str):
    if str is None:
        return True
    return len(str) == 0


def resolve_multicol(arr):
    return '_'.join(filter(lambda x: len(x) > 0 and not x == 'first', arr))


def get_inpatient_visits():
    out_fn = os.path.join(out_dir, 'visits.csv')
    cols = ['mrn', 'empi', 'account', 'patient_class_code', 'admit_date', 'admit_source_code', 'admit_type_code',
            'admit_location', 'admit_medical_service', 'discharge_date', 'discharge_status_code',
            'discharge_medical_service', 'primary_time', 'event_code', 'location_code', 'room', 'bed', 'provider_id']
    mrn_idx = cols.index('mrn')
    account_idx = cols.index('account')
    adm_idx = cols.index('admit_date')
    dis_idx = cols.index('discharge_date')
    visit_code_idx = cols.index('patient_class_code')
    df = []
    with open(visit_fn, 'r') as fd:
        overall_ct = 0
        for line in fd:
            overall_ct += 1
            if overall_ct % 1000000 == 0:
                print('Processed {} lines.  Included {} visits.'.format(overall_ct, len(df)))
            items = line.split('|')
            visit_code = _strip(items[visit_code_idx])
            if not visit_code == VISIT_TARGET_CODE:
                continue

            items = list(map(_strip, items))
            mrn = items[mrn_idx]
            account = items[account_idx]
            year = int(items[adm_idx].split('-')[0])
            assert 1900 < year < 3000  # make sure not missing other formatting
            adm_date = items[adm_idx]
            dis_date = items[dis_idx]
            should_ignore = any([
                is_none(mrn), is_none(account),
                is_none(adm_date), is_none(dis_date),
                year < MIN_YEAR, year > MAX_YEAR
            ])
            if should_ignore:
                continue
            df.append(items)
    df = pd.DataFrame(df, columns=cols)
    n1 = df.shape[0]
    print('Inpatient visits={}'.format(n1))

    df.dropna(subset=['admit_date', 'discharge_date', 'account'], inplace=True)
    n2 = df.shape[0]
    print('Removing {} null rows'.format(n1 - n2))

    df['admit_date'] = df['admit_date'].apply(str_to_dt)
    df['discharge_date'] = df['discharge_date'].apply(str_to_dt)

    df = df[df['discharge_date'] > df['admit_date']]
    n3 = df.shape[0]
    print('Removing {} rows in which discharge occurs before admit date'.format(n2 - n3))

    condensed_df = df[['account', 'mrn', 'admit_date', 'discharge_date']].groupby('account').agg({
        'mrn': ['first'],
        'admit_date': ['min', 'max'],
        'discharge_date': ['min', 'max']
    }).reset_index()
    n4 = condensed_df.shape[0]
    print('Condensing {} rows representing the same accounts'.format(n3 - n4))
    condensed_df.columns = list(map(resolve_multicol, condensed_df.columns.ravel()))
    condensed_df['admit_date_range'] = condensed_df['admit_date_max'].combine(
        condensed_df['admit_date_min'], lambda a, b: (a - b).days)
    condensed_df['discharge_date_range'] = condensed_df['discharge_date_max'].combine(
        condensed_df['discharge_date_min'], lambda a, b: (a - b).days)

    print('Saving {} rows'.format(n4))
    condensed_df.sort_values(by=['mrn', 'admit_date_min'], inplace=True)

    # Sample 1,000
    condensed_df = condensed_df.sample(n=1000, replace=False)
    condensed_df.to_csv(out_fn, index=False)

    small = condensed_df['admit_date_min'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    big = condensed_df['discharge_date_max'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    deltas = [big[i] - small[i] for i in range(len(small))]
    days = [d.total_seconds() / 86400.0 for d in deltas]

    print('Length of stay in days...')
    print(describe(days))

    return sorted(condensed_df['mrn'].unique().tolist())


if __name__ == '__main__':
    start_time = time()
    mrns = get_inpatient_visits()

    mrn_status_df = pd.DataFrame({
        'mrn': mrns,
        'has_visit': [1] * len(mrns)
    })
    mrn_status_fn = os.path.join(out_dir, 'mrn_status.csv')
    mrn_status_df.to_csv(mrn_status_fn, index=False)

    duration(start_time)

