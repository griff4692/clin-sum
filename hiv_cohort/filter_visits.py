from datetime import datetime

import pandas as pd

def str_to_dt(str):
    return datetime.strptime(str, '%Y-%m-%d-%H.%M.%S.%f')


def resolve_multicol(arr):
    return '_'.join(filter(lambda x: len(x) > 0 and not x == 'first', arr))


if __name__ == '__main__':
    in_fn = '/nlp/projects/clinsum/inpatient_visits.csv'
    out_fn = '/nlp/projects/clinsum/inpatient_visits_reduced.csv'
    df = pd.read_csv(in_fn)
    n1 = df.shape[0]
    print('Loaded inpatient visits of length {}'.format(n1))

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
    condensed_df.to_csv(out_fn, index=False)
