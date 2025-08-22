from getscore import getScore
from fusemodels import fuseModels

seeds = [1407, 2407, 3407, 4407, 5407, 6407, 7407, 8407, 9407]

score_Model0 = score_Model1 = score_Model2 = score_Model3 = score_Model4 = 0.00
for seed in seeds:
    score_Model0 += getScore('parentModel#0', seed) / len(seeds)
    score_Model1 += getScore('parentModel#1', seed) / len(seeds)
    score_Model2 += getScore('parentModel#2', seed) / len(seeds)
    score_Model3 += getScore('parentModel#3', seed) / len(seeds)
    score_Model4 += getScore('parentModel#4', seed) / len(seeds)

improvement_Model1 = score_Model1 - score_Model0
improvement_Model2 = score_Model2 - score_Model0
improvement_Model3 = score_Model3 - score_Model0
improvement_Model4 = score_Model4 - score_Model0
print(f'Mean score of parentModel#0 is {score_Model0 * 100:.4f}%, which is the baseline.')
print(f'Mean score of parentModel#1 is {score_Model1 * 100:.4f}%, a {improvement_Model1 * 100:.3f}% improvement.')
print(f'Mean score of parentModel#2 is {score_Model2 * 100:.4f}%, a {improvement_Model2 * 100:.3f}% improvement.')
print(f'Mean score of parentModel#3 is {score_Model3 * 100:.4f}%, a {improvement_Model3 * 100:.3f}% improvement.')
print(f'Mean score of parentModel#4 is {score_Model4 * 100:.4f}%, a {improvement_Model4 * 100:.3f}% improvement.')

total_improvement = improvement_Model1 + improvement_Model2 + improvement_Model3 + improvement_Model4
weight_Model1 = round(improvement_Model1 / total_improvement, 4)
weight_Model2 = round(improvement_Model2 / total_improvement, 4)
weight_Model3 = round(improvement_Model3 / total_improvement, 4)
weight_Model4 = round(improvement_Model4 / total_improvement, 4)
print(f'Assign a weight of {weight_Model1:.4f} to parentModel#1')
print(f'Assign a weight of {weight_Model2:.4f} to parentModel#2')
print(f'Assign a weight of {weight_Model3:.4f} to parentModel#3')
print(f'Assign a weight of {weight_Model4:.4f} to parentModel#4')

weights = (weight_Model1, weight_Model2, weight_Model3, weight_Model4)
fusionModel = fuseModels(weights)
score_fusionModel = 0.00
for seed in seeds: score_fusionModel += getScore(fusionModel, seed) / len(seeds)
improvement_fusionModel = score_fusionModel - score_Model0
print(f'Mean score of fusionModel is {score_fusionModel * 100:.4f}%, a {improvement_fusionModel * 100:.3f}% improvement.')
