avro_fp = '/nlp/cdw/discovery_request_1342/notes_avro/all_docs_201406190000/'
visit_fn = '/nlp/cdw/discovery_request_1342/structured/visits/visit_2004_2014.txt'
out_dir = '/nlp/projects/clinsum_2'
med_code_fn = '/nlp/cdw/note_medcodes/notetype_loinc.txt'

MIN_YEAR = 2010
MAX_YEAR = 2014
MIN_DOC_LEN = 25
NULL_STR = 'N/A'
VISIT_TARGET_CODE = 'I'

MAX_HEADER_LEN = 50
# allowable number of extra chars on either side of
# {'course', 'fen', 'gi', 'id', etc.}
# to count as acceptable hospital course (sub)header
MAX_SYSTEMS_COURSE_HEADER_LEN = 15
MIN_TARGET_LEN = 25  # min number of allowable characters in body of hospital course section
MIN_SOURCE_LEN = 100

HEADER_SEARCH_REGEX = r'(?:^|\s{4,}|\n)[\d.#]{0,4}\s*([A-Z][A-z0-9/ ]+[A-z]:|[A-Z0-9/ ]+\n)'
SEP_REGEX = r'\.\s|\n{2,}|^\s{0,}\d{1,2}\s{0,}[-).]\s{1,}'
