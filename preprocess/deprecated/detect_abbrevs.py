import pandas as pd

# import spacy
# from scispacy.abbreviation import AbbreviationDetector

# def num_abbrevs(text):
#     return min(1, len(nlp(text)._.abbreviations))


if __name__ == '__main__':
    fn = '/nlp/projects/clinsum/entity/full_entities.csv'
    entities = pd.read_csv(fn)
    sfs = set(pd.read_csv('shared_data/casi_acronyms.csv')['SF'])

    # # Add the abbreviation pipe to the spacy pipeline.
    # abbreviation_pipe = AbbreviationDetector(nlp)
    # nlp.add_pipe(abbreviation_pipe)

    source_texts = entities[entities['is_source']].sample(n=1000, replace=False)['source_value'].tolist()
    target_texts = entities[entities['is_target']].sample(n=1000, replace=False)['source_value'].tolist()

    num_source_abbrevs = len([t for t in source_texts if t in sfs])
    num_target_abbrevs = len([t for t in target_texts if t in sfs])
    print(num_source_abbrevs, num_target_abbrevs)
