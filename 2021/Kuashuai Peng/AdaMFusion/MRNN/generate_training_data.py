import numpy as np
from getscore import getScore
np.random.seed(3407)

centre_weight = (0.0, 0.254, 0.343, 0.169, 0.234)
sigma = 0.2
datasize = 1000
distribution_w1 = np.random.normal(centre_weight[1], sigma, datasize)
distribution_w2 = np.random.normal(centre_weight[2], sigma, datasize)
distribution_w3 = np.random.normal(centre_weight[3], sigma, datasize)
distribution_w4 = np.random.normal(centre_weight[4], sigma, datasize)


def isFeasible(weights):
    feasibility = True
    weights = weights[1:]
    for w in weights:
        if w < 0 or w > 1:
            feasibility = False
    return feasibility


with open('MRNN_training_data.txt', 'w') as f:
    f.write('w_0,w_1,w_2,w_3,w_4,accuracy,precision,recall,f1-score\n')
count = 0
for i in range(datasize):
    if count == 250:
        break
    w_1 = round(distribution_w1[i], 5)
    w_2 = round(distribution_w2[i], 5)
    w_3 = round(distribution_w3[i], 5)
    w_4 = round(distribution_w4[i], 5)
    w_0 = round(1 - w_1 - w_2 - w_3 - w_4, 5)
    weights = (w_0, w_1, w_2, w_3, w_4)
    if isFeasible(weights):
        count += 1
        score = getScore(weights)
        print(f'No.{count}\t weights{weights} has a score of {score}')
        with open('MRNN_training_data.txt', 'a') as f:
            print(f'{weights[0]},{weights[1]},{weights[2]},{weights[3]},{weights[4]},'
                  f'{score[0]},{score[1]},{score[2]},{score[3]}', file=f)
