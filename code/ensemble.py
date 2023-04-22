import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import torch.nn.functional as F
import torch

# ML 내부에 inferences 라는 폴더 생성해 주시고, 해당 inferences 폴더 내부에 앙상블 하고자 하는 output.csv 파일을 넣으면
# 자동으로 긁어와서 ensemble_output.csv 파일을 생성하게 됩니다.
# *주의* inferences 폴더 내부의 파일은 <파일이름_score.csv> 형식을 지켜주셔야 합니다.
# ex) inferences/output_0.97.csv
# '_' 를 기준으로 filename 과 score 를 인식하기 때문에 언더바 사용에 주의해주세요
# inferences 폴더와 내부 파일 생성 후에는 그냥 python code/ensemble.py 로 작동하시면 됩니다~!

class Ensemble():
    def __init__(self):
        self.files = os.listdir('./inferences')
        self.files = [(file,float(file.replace('.csv',"").split('_')[1])) for file in self.files]
        
    def soft_vote_ensemble(self):
        num_files = len(self.files) #(filename, score)
        scores = torch.Tensor([inference[1] for inference in self.files])
        inf_list = [pd.read_csv('./inferences/'+inference[0])['target'] for inference in self.files]

        scores = F.softmax(scores, dim=-1)
        inf_list = [inf_list[i]*scores[i].item() for i in range(num_files)]
        concatenated_inf = pd.concat(inf_list, axis=1)
        ensemble_output = pd.Series(concatenated_inf.sum(axis = 1))
        
        for i in range(len(ensemble_output)):
            if ensemble_output.iloc[i] > 5:
                ensemble_output.iloc[i] = 5
            elif ensemble_output.iloc[i] < 0:
                ensemble_output.iloc[i] = 0

        output = pd.read_csv('./data/sample_submission.csv')
        output['target'] = ensemble_output
        output.to_csv('[last3]ensemble_output.csv', index=False)


#e = Ensemble([('output.csv', 0.5), ('output.csv', 0.8), ('output.csv', 0.7)])
e = Ensemble()
e.soft_vote_ensemble()