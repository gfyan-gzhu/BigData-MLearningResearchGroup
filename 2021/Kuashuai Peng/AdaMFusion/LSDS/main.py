from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification
import collections
from datetime import datetime, timedelta
from getscore import getScore
from searching import forwardSearch, backwardSearch
from transformers import logging
logging.set_verbosity_error()


step_size = 0.1
step_size_reduction_coefficient = 0.75
MIN_step_size = _

exec_time = timedelta(minutes=0)
MAX_exec_time = timedelta(hours=_)

w_0 = 1.0
w_1 = w_2 = w_3 = w_4 = 0.0
w_Opt = w_Int = (w_0, w_1, w_2, w_3, w_4)

C_0 = AutoModelForMaskedLM.from_pretrained('./parentModel#0/')
C_1 = AutoModelForMaskedLM.from_pretrained('./parentModel#1/')
C_2 = AutoModelForMaskedLM.from_pretrained('./parentModel#2/')
C_3 = AutoModelForSequenceClassification.from_pretrained('./parentModel#3/', num_labels=10)
C_4 = AutoModelForSequenceClassification.from_pretrained('./parentModel#4/', num_labels=5)
C_Set = (C_0, C_1, C_2, C_3, C_4)

with open('getScore_log.txt', 'w') as f:
    f.write('w_0,w_1,w_2,w_3,w_4,score\n')

print(f'LSDS algorithm starts running, where Δ={step_size:.2f} & γ={step_size_reduction_coefficient:.2f}')
print('>>>>------------------------------------------------')
scoresDB = collections.OrderedDict()
scoresDB[w_Int] = getScore(w_Int, C_Set, scoresDB)
loop_count = reduction_times = 0
start_time = datetime.now()
while step_size >= MIN_step_size and exec_time <= MAX_exec_time:
    w_Opt_F = forwardSearch(w_Int, C_Set, step_size, scoresDB)
    w_Opt_B = backwardSearch(w_Int, C_Set, step_size, scoresDB)
    if getScore(w_Opt_F, C_Set, scoresDB) > getScore(w_Opt_B, C_Set, scoresDB):
        w_Opt = w_Opt_F
    else:
        w_Opt = w_Opt_B
    loop_count += 1
    print(f'>> Loop {loop_count} determines w_Opt = <{w_Opt[0]}, {w_Opt[1]}, {w_Opt[2]}, {w_Opt[3]}, {w_Opt[4]}>')
    exec_time = datetime.now() - start_time
    print(f'// LSDS has been running for {exec_time.seconds // 60} minutes.')
    print('----------' * 7)
    if w_Opt == w_Int:
        reduction_times += 1
        previous_step_size = step_size
        step_size *= step_size_reduction_coefficient
        step_size = round(step_size, 6)
        print(f'\u21B3\u21B3 No.{reduction_times} step size reduction, Δ from {previous_step_size} to {step_size}')
        print('----------' * 7)
    w_Int = w_Opt

print('**********' * 7)
print(f'LSDS finally determines w_Opt = <{w_Opt[0]}, {w_Opt[1]}, {w_Opt[2]}, {w_Opt[3]}, {w_Opt[4]}>')
