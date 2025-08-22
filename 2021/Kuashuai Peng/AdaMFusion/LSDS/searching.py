import copy
from getscore import getScore

masks = [
    [[0, 0, 0, 0]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
    [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]],
    [[1, 1, 1, 1]]
]


def isFeasible(soln):
    feasibility = True
    for w in soln:
        if w < 0 or w > 1:
            feasibility = False
    return feasibility


def findAllcases(w_Int, n, d, step_size):
    Allcases = []
    for mask in masks[n]:
        w_temp = copy.deepcopy(w_Int)
        for i, flag in enumerate(mask):
            if flag:
                if d == 'F': w_temp[i + 1] += step_size
                if d == 'B': w_temp[i + 1] -= step_size
                w_temp[i + 1] = round(w_temp[i + 1], 6)
        if isFeasible(w_temp):
            Allcases.append(tuple(w_temp))
    return Allcases


def forwardSearch(w_Int, C_Set, step_size, scoresDB):
    w_Opt_F = w_Int
    Opt_score = getScore(w_Opt_F, C_Set, scoresDB)
    w_Int_F = list(w_Opt_F)
    n = 1
    while w_Int_F[0] - step_size >= 0 and n <= 4:
        w_Int_F[0] -= step_size
        w_Int_F[0] = round(w_Int_F[0], 6)
        Allcases = findAllcases(w_Int_F, n, 'F', step_size)
        for case in Allcases:
            current_score = getScore(case, C_Set, scoresDB)
            if current_score > Opt_score:
                w_Opt_F = case
                Opt_score = getScore(w_Opt_F, C_Set, scoresDB)
        if w_Opt_F != w_Int:
            break
        n += 1
    return w_Opt_F


def backwardSearch(w_Int, C_Set, step_size, scoresDB):
    w_Opt_B = w_Int
    Opt_score = getScore(w_Opt_B, C_Set, scoresDB)
    w_Int_B = list(w_Opt_B)
    n = 1
    while w_Int_B[0] + step_size <= 1 and n <= 4:
        w_Int_B[0] += step_size
        w_Int_B[0] = round(w_Int_B[0], 6)
        Allcases = findAllcases(w_Int_B, n, 'B', step_size)
        for case in Allcases:
            current_score = getScore(case, C_Set, scoresDB)
            if current_score > Opt_score:
                w_Opt_B = case
                Opt_score = getScore(w_Opt_B, C_Set, scoresDB)
        if w_Opt_B != w_Int:
            break
        n += 1
    return w_Opt_B
