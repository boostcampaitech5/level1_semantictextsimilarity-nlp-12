import numpy as np
import pandas as pd
from konlpy.tag import Okt
from koeda import EDA
import copy
import os


def zero_to_five(zero_df):
    okt = Okt()
    check_pos = ['NounNoun', 'AdjectiveNoun',
                 'DeterminerNoun', 'VerbNoun', 'AdverbVerb']
    add_func = [".", ",", "!"]

    tmp = pd.DataFrame(columns=csv_df.columns)
    sample_id = 9324

    for src, row in zip(zero_df['source'], zero_df['sentence_1']):
        sample_id += 1
        source = src.replace('sampled', 'rtt')
        text = copy.deepcopy(row)

        tokenized_row = okt.pos(text)
        origin_str = None
        replace_str = None
        for idx, token in enumerate(tokenized_row):
            if idx > 0:
                if ''.join([tokenized_row[idx][1], tokenized_row[idx - 1][1]]) in check_pos:
                    if row.find(' '.join([tokenized_row[idx - 1][0], tokenized_row[idx][0]])) != -1:
                        origin_str = ' '.join(
                            [tokenized_row[idx - 1][0], tokenized_row[idx][0]])
                        replace_str = ''.join(
                            [tokenized_row[idx - 1][0], tokenized_row[idx][0]])
                        break

        if (origin_str != None and replace_str != None):
            text = text.replace(origin_str, replace_str, 1)
        else:
            text = text + add_func[sample_id % 3]

        ###################################################
        print("id : ", sample_id, " <->  sentence : ", text)
        dic = {
            'id': 'boostcamp-sts-v1-train-' + str(sample_id - 91),
            'source': source,
            'sentence_1': row,
            'sentence_2': text,
            'label': 5.0,
            'binary-label': 1
        }
        dic_df = pd.DataFrame(dic, index=[sample_id])
        tmp = pd.concat([tmp, dic_df], axis=0)
    tmp.to_csv('only_five.csv')
    return tmp  # label 0 짜리 1000개  label 5짜리 1000개로 변환


def aug_label_four(df, repeat, label, label_under_bar):
    eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.05)
    
    tmp = pd.DataFrame(columns=data.columns)
    sample_id = len(data)
    
    for source, s1, s2, bi_label in zip(df['source'], df['sentence_1'], df['sentence_2'], df['binary-label']): # id,source,sentence_1,sentence_2,label,binary-label
        if sample_id >= sample_id - len(data) >= repeat:
            break

        sample_id += 1
        source = source
        binary_label = bi_label
        text1, text2 = eda([s1, s2]) 
        ###################################################
        print("original : ", s1, " ------- ", s2)
        print("id : ", sample_id, " <->  sentence : ", "[",text1,' ------- ',text2, "]")
        dic = {
            'id': 'boostcamp-sts-v1-train-' + str(sample_id+len(data)),
            'source': source,
            'sentence_1': text1,
            'sentence_2': text2,
            'label': label,
            'binary-label': binary_label
        }
        dic_df = pd.DataFrame(dic, index=[sample_id])
        tmp = pd.concat([tmp, dic_df], axis=0)
    tmp.to_csv('aug_four_' + label_under_bar + '.csv')

    return tmp  # label 0 짜리 1000개  label 5짜리 1000개로 변환

###########################################################
#main
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-20\bin\server' # jdk path your coumputer
print("Connect : ", 'JAVA_HOME' in os.environ)
path = './'
csv_df = pd.read_csv("./train.csv")

# ===> label zero cut
zero_labels = csv_df[csv_df["label"] == 0]
not_zero = csv_df[csv_df["label"] != 0]
data = pd.concat([zero_labels[:1000], not_zero], axis = 0)
# data.to_csv('d8_zero_200.csv', float_format='%.1f')

# ===> label five cut
df_five = zero_to_five(zero_labels[:250])
data = pd.concat([data, df_five], axis=0)

# ==> EDA
aug_four_six = data[data['label'] == 4.6]
df_fx = aug_label_four(aug_four_six[:30],30, 4.6, "4_6")
data = pd.concat([data, aug_four_six], axis=0)
aug_four_eight = data[data['label'] == 4.8]
df_fe = aug_label_four(aug_four_eight[:20] ,20, 4.8, "4_8")
data = pd.concat([data, aug_four_eight], axis=0)
print("cols : ", sorted(data['label'].unique()))

# label five to csv
data.to_csv('d13_final_train.csv', float_format='%.1f')
###########################################################