#!/usr/bin/env python3

from fuzzywuzzy import process
from Levenshtein import ratio as lev_ratio

lev_ratio = Levenshtein.lev_ratio('ab', 'a')
fuzz_extract = process.extract(word, infer_txts_list, scorer=lev_ratio, limit=10)