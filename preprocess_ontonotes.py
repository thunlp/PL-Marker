import conll
import collections
import re
import json
counter = collections.Counter()
def normalize_word(word, language='english'):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def get_original_token(token):
    escape_to_original = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    if token in escape_to_original:
        token = escape_to_original[token]
    return token
    
def prosess(prefix):
    input_path = prefix + '.english.v4_gold_conll'
    documents = []
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            begin_document_match = re.match(conll.BEGIN_DOCUMENT_REGEX, line)
            if begin_document_match:
                doc_key = conll.get_doc_key(begin_document_match.group(1), begin_document_match.group(2))
                documents.append((doc_key, []))
            elif line.startswith("#end document"):
                continue
            else:
                documents[-1][1].append(line)

    output_w = open(prefix + '.json', 'w')
    skip_doc = 0
    for document_lines in documents:
        doc_key = document_lines[0]

        sents = []
        ners = []
        sent = []
        ner = []
        word_idx = 0
        last_word_idx = -1
        ner_type = None
        for line in document_lines[1]:
            tok_info = line.strip().split()
            if len(tok_info) == 0:
                assert (last_word_idx==-1)
                if len(sent) > 0:
                    sents.append(sent)
                    ners.append(ner)
                    sent = []
                    ner = []
                    continue

            word = get_original_token(tok_info[3])
            word = normalize_word(word)

            label = tok_info[10] if (tok_info is not None and len(tok_info)>0) else '-'
            if label != "*":
                if label[0] == "(":
                    ner_type = label[1:-1]
                    if label[-1] == ')':
                        ner.append( (word_idx, word_idx, ner_type) )                         
                    else:
                        last_word_idx = word_idx
                elif label=='*)':
                    ner.append( (last_word_idx, word_idx, ner_type) ) 
                    counter[ner_type] += 1
                    last_word_idx = -1
                else:
                    assert(False)


            sent.append(word)
            word_idx += 1

        if doc_key.startswith('pt/'):
            tot_ner = 0
            for ner in ners:
                tot_ner += len(ner)
            assert(tot_ner == 0)
            skip_doc += 1
            continue

        item = {'sentences': sents,
                'ner': ners,
                'doc_key': doc_key
            }

        output_w.write(json.dumps(item)+'\n')
    print (prefix, 'skip doc:', skip_doc)

data_dir = 'ontonotes/'
prosess(data_dir + 'dev')
prosess(data_dir + 'test')
prosess(data_dir + 'train')

print (counter)
print (sorted(list(counter.keys())))