#read train.txt and test.txt

import codecs

def map_tags_to_numbers(fileName):
    
    mapping = {}
    with codecs.open(fileName,  encoding='utf-8') as f:
        for line in f:
            if len(line) <2:
                continue
            if line.split(' ')[3][:-1] not in mapping:
                mapping[line.split(' ')[3][:-1]] = str(len(mapping))

    return mapping

def process_data(fileName, writeTo):
    count = 0

    with codecs.open(writeTo,  encoding='utf-8', mode= 'w') as wf:
        with codecs.open(fileName,  encoding='utf-8') as f:
            tokens = []
            ner_tags = []

            for line in f:
                if len(line) <2:
                    
                    # if "\"B-GRP\"" in ner_tags or"\"I-GRP\"" in ner_tags or '\"B-CORP\"' in ner_tags or '\"I-CORP\"' in ner_tags or "\"B-Medical\"" in ner_tags or "\"I-Medical\"" in ner_tags:
                    #     tokens = []
                    #     ner_tags = []
                    #     continue
                    
                    wr = "{\"tokens\":["+ ','.join(tokens) + '], \"tags\":[' + ','.join(ner_tags) + ']}'
                    wf.write(wr)
                    wf.write('\n')
                    tokens = []
                    ner_tags = []
                    count += 1
                    continue

                tokens.append('\"'+line.split(' ')[0]+'\"')
                
                ner_tags.append('\"'+line.split(' ')[3][:-1]+'\"')
                
            wr = "{\"tokens\":["+ ','.join(tokens) + '], \"tags\":[' + ','.join(ner_tags) + ']}'
                   
            wf.write(wr)
            wf.write('\n')
                    
    print(count)

def process_data_oversample(fileName, writeTo):
    count = 0

    with codecs.open(writeTo,  encoding='utf-8', mode= 'w') as wf:
        with codecs.open(fileName,  encoding='utf-8') as f:
            tokens = []
            ner_tags = []

            for line in f:
                if len(line) <2:
                    
                    wr = "{\"tokens\":["+ ','.join(tokens) + '], \"tags\":[' + ','.join(ner_tags) + ']}'
                    if "\"B-CW\"" in ner_tags or '\"B-PROD\"' in ner_tags:
                        wf.write(wr)
                        wf.write('\n')
                        count += 1
                    
                    wf.write(wr)
                    wf.write('\n')
                    tokens = []
                    ner_tags = []
                    count += 1
                    continue

                tokens.append('\"'+line.split(' ')[0]+'\"')
                
                ner_tags.append('\"'+line.split(' ')[3][:-1]+'\"')
                
            wr = "{\"tokens\":["+ ','.join(tokens) + '], \"tags\":[' + ','.join(ner_tags) + ']}'
                   
            wf.write(wr)
            wf.write('\n')
                    
    print(count)

def process_data_2023(fileName, writeTo):
    count = 0

    with codecs.open(writeTo,  encoding='utf-8', mode= 'w') as wf:
        with codecs.open(fileName,  encoding='utf-8') as f:
            tokens = []
            ner_tags = []

            for line in f:
                if len(line) <2:
                    
                    if "\"B-GRP\"" in ner_tags or"\"I-GRP\"" in ner_tags or '\"B-CORP\"' in ner_tags or '\"I-CORP\"' in ner_tags or "\"B-Medical\"" in ner_tags or "\"I-Medical\"" in ner_tags:
                        tokens = []
                        ner_tags = []
                        continue
                    
                    wr = "{\"tokens\":["+ ','.join(tokens) + '], \"tags\":[' + ','.join(ner_tags) + ']}'
                    wf.write(wr)
                    wf.write('\n')
                    tokens = []
                    ner_tags = []
                    count += 1
                    continue

                tokens.append('\"'+line.split(' ')[0]+'\"')
                
                ner_tags.append('\"'+line.split(' ')[3][:-1]+'\"')
                
            wr = "{\"tokens\":["+ ','.join(tokens) + '], \"tags\":[' + ','.join(ner_tags) + ']}'
                   
            wf.write(wr)
            wf.write('\n')
                    
    print(count)

'''
Required format:
{"tokens":["সবচেয়ে","সাম্প্রতিক","বিজয়ী","হল","জ্যাক","গ্রিনকে","।"], "tags":["O","O","O","O","B-PER","I-PER","O"]}
{"tokens":["এটি","ইকুয়েডর","(নাপো","প্রদেশ)","থেকে","স্থানীয়।"], "tags":["O","B-LOC","O","O","O","O"]}
{"tokens":["বাষ্প","প্রতিস্থাপিত","হয়েছে","ব্রিটিশ","রেল","ক্লাস","৩৭","দ্বারা,","জোড়ায়","কাজ","করে।"], "tags":["O","O","O","B-PROD","I-PROD","I-PROD","I-PROD","O","O","O","O"]}
{"tokens":["যাইহোক,","ইউএসএ","হকি","জাতীয়","দল","উন্নয়ন","কর্মসূচি","অপেশাদারদের","দ্বারা","গঠিত,","কলেজ","স্কোয়াডগুলিকে","তাদের","খেলতে","দেয়।",")"], "tags":["O","B-GRP","I-GRP","I-GRP","I-GRP","I-GRP","I-GRP","O","O","O","O","O","O","O","O","O"]}
{"tokens":["১৯৬৫","সালে","কোম্পানিটি","জাতীয়করণ","করা","হয়েছিল","এবং","রেনফি","ফেভ","এ","রূপান্তরিত","হয়েছিল,","এর","পরে","স্টেশন","ভবনটি","জনসাধারণের","হাতে","চলে","যায়।"], "tags":["O","O","O","O","O","O","O","B-CORP","I-CORP","O","O","O","O","O","O","O","O","O","O","O"]}
{"tokens":["রিডকে","পেঙ্গুইন","বুক্স","থেকে","ক্লাসিক","হরর","উপন্যাসের","গ্রাফিক","নভেল","অ্যাডাপ্টেশন","লেখার","জন্য","আনা","হয়েছিল।"], "tags":["O","B-CORP","I-CORP","O","O","O","O","O","O","O","O","O","O","O"]}
{"tokens":["১৯৭৮","সালে,","তিনি","বলেছিলেন","যে","তিনি","১৫","বছরে","রেস্তোরাঁর","খাবার","খাননি,","এবং","শুধুমাত্র","জৈব","খাদ্য","পছন্দ","করেন।"], "tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","B-PROD","I-PROD","O","O"]}
'''

#Some preprocessing like replacing " with \" and removing invalid escape characters were done manually
process_data("train.txt", "processed_train.jsonl")
process_data("dev.txt", "processed_dev.jsonl")
process_data_2023("bn-train.conll", "processed_2023.jsonl")

#processed_2023.jsonl was appended to processed_train.jsonl