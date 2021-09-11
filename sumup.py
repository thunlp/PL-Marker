import json
import sys
import numpy as np


if sys.argv[1].endswith('ner'):
    delta = 0

    if sys.argv[1].startswith('sci'):
        prefix = 'sciner_models/'+sys.argv[2]
    elif sys.argv[1].startswith('ace04'):
        prefix = 'ace04ner_models/'+sys.argv[2]
        delta = 42
    elif sys.argv[1].startswith('conll03'):
        prefix = 'conll03_models/'+sys.argv[2]
    elif sys.argv[1].startswith('fewnerd'):
        prefix = 'fewnerd_models/'+sys.argv[2]
    elif sys.argv[1].startswith('ontonotes'):
        prefix = 'ontonotes_models/'+sys.argv[2]
    else:
        prefix = 'ace05ner_models/'+sys.argv[2]

    f1s = []
    for i in range(42-delta, 47-delta):
        fileename = prefix + '-' + str(i) + '/results.json'
        try:
            f1 = json.load(open(fileename))['f1_overlap_']
            print (fileename)
            f1s.append(f1)
        except:
            pass
    try:
        print ('F1_overlap:')
        print (f1s)
        print (sum(f1s)/len(f1s))
    except:
        pass


    precisions = []
    for i in range(42-delta, 47-delta):
        filename = prefix + '-' + str(i) + '/results.json'
        try:
            precision = json.load(open(filename))['precision_']
            print (filename)
            precisions.append(precision)
        except:
            pass
    try:
        print ('Precision:')
        print (precisions)
        precisions = np.array(precisions)
        print (np.mean(precisions)*100)
        print (np.std(precisions)*100)
    except:
        pass

    recalls = []
    for i in range(42-delta, 47-delta):
        filename = prefix + '-' + str(i) + '/results.json'
        try:
            try:
                recall = json.load(open(filename))['recall_score_']
            except:
                recall = json.load(open(filename))['recall_']

            print (filename)
            recalls.append(recall)
        except:
            pass

    try:
        print ('Recall:')
        print (recalls)
        recalls = np.array(recalls)
        print (np.mean(recalls)*100)
        print (np.std(recalls)*100)

    except:
        pass


    f1s = []
    for i in range(42-delta, 47-delta):
        filename = prefix + '-' + str(i) + '/results.json'
        try:
            f1 = json.load(open(filename))['f1_']
            print (filename)
            f1s.append(f1)
        except:
            pass

    try:
        print ('F1:')
        print (f1s)
        f1s = np.array(f1s)
        print (np.mean(f1s)*100)
        print (np.std(f1s)*100)
    except:
        pass

elif  sys.argv[1].endswith('re'):
    delta = 0
    if sys.argv[1].startswith('sci'):
        prefix = 'scire_models/'+sys.argv[2]
    elif sys.argv[1].startswith('ace04'):
        prefix = 'ace04re_models/'+sys.argv[2]
        delta = 42
    else:
        prefix = 'ace05re_models/'+sys.argv[2]

    resultfilename = '/results.json'
    f1s = []
    for i in range(42-delta, 47-delta):
        fileename = prefix + '-' + str(i) + resultfilename
        try:
            f1 = json.load(open(fileename))['ner_f1_']
            print (fileename)
            f1s.append(f1)
        except:
            pass

    try:
        print ('NER F1:')
        print (f1s)
        # print (sum(f1s)/len(f1s))
        f1s = np.array(f1s)
        print (np.mean(f1s)*100)
        print (np.std(f1s)*100)

    except:
        pass


    f1s = []
    for i in range(42-delta, 47-delta):
        fileename = prefix + '-' + str(i) + resultfilename
        try:
            f1 = json.load(open(fileename))['f1_']
            print (fileename)
            f1s.append(f1)
        except:
            pass

    print ('F1:')
    print (f1s)

    f1s = np.array(f1s)
    print (np.mean(f1s)*100)
    print (np.std(f1s)*100)

    f1s = []
    for i in range(42-delta, 47-delta):
        fileename = prefix + '-' + str(i) + resultfilename
        try:
            f1 = json.load(open(fileename))['f1_with_ner_']
            print (fileename)
            f1s.append(f1)
        except:
            pass

    print ('F1+:')
    print (f1s)

    f1s = np.array(f1s)
    print (np.mean(f1s)*100)
    print (np.std(f1s)*100)


