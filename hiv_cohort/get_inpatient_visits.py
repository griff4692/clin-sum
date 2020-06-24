from collections import defaultdict

import pandas as pd


def _strip(str):
    return str.strip(' "\n')


if __name__ == '__main__':
    TARGET_CODE = 'I'

    type_ct = defaultdict(int)

    visit_fn = '/nlp/cdw/discovery_request_1342/structured/visits/visit_2004_2014.txt'
    out_fn = '/nlp/projects/clinsum/inpatient_visits.csv'

    cols = ['mrn', 'empi', 'account', 'patient_class_code',  'admit_date', 'admit_source_code', 'admit_type_code',
            'admit_location', 'admit_medical_service', 'discharge_date', 'discharge_status_code',
            'discharge_medical_service', 'primary_time', 'event_code', 'location_code', 'room', 'bed', 'provider_id']
    df = []
    with open(visit_fn, 'r') as fd:
        overall_ct = 0
        for line in fd:
            items = line.split('|')
            visit_code = _strip(items[3])
            year = int(_strip(items[4]).split('-')[0])
            if year < 2010 or year > 2014:
                continue
            type_ct[visit_code] += 1
            if visit_code == TARGET_CODE:
                df.append(list(map(_strip, items)))
            overall_ct += 1
            if overall_ct % 1000000 == 0:
                print('Processed {} lines.'.format(overall_ct))
    print('Patient Code Counts:')
    codes = sorted(type_ct.keys())
    for code in codes:
        print('\t{} --> {}'.format(code, type_ct[code]))
    df = pd.DataFrame(df, columns=cols)
    print('Saving {} rows of inpatient visits to {}...'.format(df.shape[0], out_fn))
    df.to_csv(out_fn, index=False)
