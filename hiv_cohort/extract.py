from collections import defaultdict
from datetime import timedelta
import os
import pickle

import pandas as pd

# CONSTANTS
DATA_DIR = '/nlp/projects/hiv_clinic_phenotyping/omop_data/output/cohort_data_files/'
MED_CODE_FN = '/nlp/projects/oliver_avroProcessing/medCodeMapping/dataSaves/medCodeMetaData.csv'


def dump(arr, cols, fn):
    df = pd.DataFrame(arr, columns=cols)
    df.drop(columns=['person_id.1'], inplace=True)
    df.to_csv(fn, index=False)


def get_med_code_info(med_code_df, med_code):
    info = defaultdict(list)
    med_code_row = med_code_df[med_code_df['medCode'] == med_code]
    if med_code_row.shape[0] > 0:
        med_code_dict = med_code_row.iloc[0].to_dict()

        for k, v in med_code_dict.items():
            if k == 'medCode' or v == 0:
                continue
            prefix = k.split('_')[0]
            info[prefix].append(k)
    return info


if __name__ == '__main__':
    #############
    # note data
    #############
    notes_fn = os.path.join(DATA_DIR, 'hiv_all_notes_w_dups.pickle')
    with open(notes_fn, 'rb') as fd:
        notes = pickle.load(fd)
    notes.columns = map(str.lower, notes.columns)

    mrn_person_id = os.path.join(DATA_DIR, 'mrn_person_id_list.pickle')
    with open(mrn_person_id, 'rb') as fd:
        id_lookup = pickle.load(fd)

    ################################
    #merge person id to notes data##
    ################################

    notes = notes.merge(id_lookup, on='mrn', how='inner')

    ####################
    # transform dates ##
    #####################
    notes = notes[['person_id', 'mrn', 'time_str_key', 'primary_time', 'update_time', 'event_code', 'title', 'text']]
    dup_notes_i = notes.duplicated(subset=['mrn', 'primary_time', 'title'] , keep=False)
    notes = notes[~dup_notes_i]
    notes['time_str_key'] = pd.to_datetime(notes.time_str_key, format='%Y-%m-%d-%H.%M.%S.%f', errors='coerce')
    notes['primary_time'] = pd.to_datetime(notes.primary_time, unit='ms', origin='unix', errors='coerce')
    notes['update_time'] = pd.to_datetime(notes.update_time, unit='ms', origin='unix', errors='coerce')
    notes['date'] = notes['time_str_key'].dt.date
    note_cols = list(notes.columns)
    ####################
    #read visit data ##
    #####################

    #* patient_class_code - values are from the following:
    #"A"  ambulatory surgery
    #"C"  clinic visit
    #"D"  DPO visits
    #"E"  ER visits
    #"I"   Inpatient hospitalizations
    #"O"  outpatient visits
    #"P"   preadmit testing
    #"R"  recurring visits (DPO)
    # --> in df it is visit_source_value
    med_code_cols = ['setting', 'author', 'subject', 'service']
    med_code_df = pd.read_csv(MED_CODE_FN)

    visit_fn = os.path.join(DATA_DIR, 'hiv_visit_occurrence.pickle')
    with open(visit_fn, 'rb') as fd:
        vst = pickle.load(fd)

    imp_cols = ['visit_occurrence_id', 'person_id', 'visit_concept_id', 'visit_start_date', 'visit_start_datetime',
                'visit_end_date', 'visit_end_datetime', 'visit_source_value']
    vst_sub = vst[imp_cols]
    vst_sub.dropna(how='any', inplace=True)
    vst_sub = vst_sub.sort_values(by=['person_id', 'visit_start_date', 'visit_end_date'])

    vst_cols = list(vst_sub.columns)

    augmented_df = []
    augmented_cols = med_code_cols + note_cols + vst_cols
    ct = 0
    curr_person = None
    curr_person_df = None
    vst_n = vst_sub.shape[0]
    for idx, vst_row in vst_sub.iterrows():
        vst_row = vst_row.to_dict()
        visit_occurrence_id = vst_row['visit_occurrence_id']
        person_id = vst_row['person_id']

        if curr_person is None or not person_id == curr_person:
            curr_person_df = notes[notes['person_id'] == person_id]
            curr_person = person_id

        visit_start_date = pd.Timestamp(vst_row['visit_start_date'])
        visit_end_date = pd.Timestamp(vst_row['visit_end_date'] + timedelta(days=1))

        visit_start_datetime = vst_row['visit_start_datetime']
        visit_end_datetime = vst_row['visit_end_datetime']

        matching_notes = curr_person_df[
            (notes['primary_time'] >= visit_start_datetime)
            & (notes['primary_time'] < visit_end_datetime)]

        for j, note_row in matching_notes.iterrows():
            note_row = note_row.to_dict()
            med_code_info = get_med_code_info(med_code_df, note_row['event_code'])

            augmented_row = []
            if len(med_code_info) == 0:
                augmented_row += [None] * len(med_code_cols)
            else:
                for k in med_code_cols:
                    augmented_row.append('|'.join(med_code_info[k.upper()]))
            for note_col in note_cols:
                augmented_row.append(note_row[note_col])
            for vst_col in vst_cols:
                augmented_row.append(vst_row[vst_col])
            augmented_df.append(augmented_row)
        ct += 1
        if ct % 1000 == 0:
            print('Processed {} out of {} visits'.format(ct, vst_n))
            if ct == 10000:
                dump(augmented_df, augmented_cols, 'data/mini_notes.csv')
    dump(augmented_df, augmented_cols, 'data/mini_notes.csv')
