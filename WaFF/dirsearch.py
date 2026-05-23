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
    for w in soln:
        if w < 0 or w > 1:
            return False
    return True


def findAllcases(w_Int, n, d, step_size):
    Allcases = []
    for mask in masks[n]:
        w_temp = copy.deepcopy(w_Int)
        for i, flag in enumerate(mask):
            if flag:
                if d == 'F':
                    w_temp[i + 1] += step_size
                if d == 'B':
                    w_temp[i + 1] -= step_size
                w_temp[i + 1] = round(w_temp[i + 1], 6)
        if isFeasible(w_temp):
            Allcases.append(tuple(w_temp))
    return Allcases


def searchDirectionally(w_Int, C_Set, step_size, scoresDB, direction):
    w_Opt = w_Int
    Opt_score = getScore(w_Opt, C_Set, scoresDB)
    w_Int_curr = list(w_Opt)
    n = 1
    def boundaryCheck(value):
        if direction == 'F':
            return value - step_size >= 0
        else:
            return value + step_size <= 1
    while boundaryCheck(w_Int_curr[0]) and n <= 4:
        if direction == 'F':
            w_Int_curr[0] -= step_size
        else:
            w_Int_curr[0] += step_size
        w_Int_curr[0] = round(w_Int_curr[0], 6)
        Allcases = findAllcases(w_Int_curr, n, direction, step_size)
        for case in Allcases:
            current_score = getScore(case, C_Set, scoresDB)
            if current_score > Opt_score:
                w_Opt = case
                Opt_score = current_score
        if w_Opt != w_Int:
            break
        n += 1
    return w_Opt
