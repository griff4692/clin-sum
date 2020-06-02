from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd


def kvprint(k, v):
    print('{}={}'.format(k, v))


def value_counts(df, col):
    return df.groupby([col]).size().reset_index(name='counts')


if __name__ == '__main__':
    df = pd.read_csv('data/notes.csv')
    df.drop(columns=['person_id.1'], inplace=True)

    other_dup_cols = ['mrn', 'title', 'time_str_key']
    dup_cols = ['visit_occurrence_id', 'text']
    df = df[~df.duplicated(subset=dup_cols)]
    df = df[~df.duplicated(subset=other_dup_cols)]

    N = df.shape[0]
    df['is_discharge'] = df['title'].apply(lambda x: 'discharge' in x.lower())
    mrns = df['mrn'].unique().tolist()

    visit_note_counts = value_counts(df, 'visit_occurrence_id')
    kvprint('Visit Count', visit_note_counts.shape[0])
    kvprint('MRN Count', len(mrns))
    kvprint('Note Count', N)

    sd_visits_df = visit_note_counts[visit_note_counts['counts'] == 1]
    md_visits_df = visit_note_counts[visit_note_counts['counts'] > 1]

    # Histogram
    ax = md_visits_df[['counts']].plot.hist(by='counts', bins=99, range=(2, 101))
    visit_counts_hist_fn = 'tmp.png'
    ax.figure.savefig(visit_counts_hist_fn)
    print('Saved visit counts histogram to {}'.format(visit_counts_hist_fn))

    num_visits_sd = sd_visits_df.shape[0]
    num_visits_md = md_visits_df.shape[0]
    print('SD Visit Count', num_visits_sd)
    print('MD Visit Count', num_visits_md)
    md_notes_df = df[df['visit_occurrence_id'].isin(md_visits_df['visit_occurrence_id'])]

    md_mrns = md_notes_df['mrn'].unique().tolist()
    kvprint('MD MRN Count', len(md_mrns))
    kvprint('MD Note Count', md_notes_df.shape[0])

    discharge_df = md_notes_df[md_notes_df['is_discharge']]
    mds_visit_w_discharge_ids = discharge_df['visit_occurrence_id'].unique().tolist()
    kvprint('DS MD Visit Count', len(mds_visit_w_discharge_ids))
    discharge_md_mrns = discharge_df['mrn'].unique().tolist()
    kvprint('DS MD MRN Count', len(discharge_md_mrns))

    final_df = df[df['visit_occurrence_id'].isin(mds_visit_w_discharge_ids)]
    kvprint('DS MD Note Count', final_df.shape[0])
    # for i, id in enumerate(mds_visit_w_discharge_ids):
    #     x = discharge_df[discharge_df['visit_occurrence_id'] == id]
    #     num_unique_texts = set(x['text'].tolist())
    #     print(len(num_unique_texts), x.shape[0])
    #     print('\n\n\n')
    #     if i > 10:
    #         break
