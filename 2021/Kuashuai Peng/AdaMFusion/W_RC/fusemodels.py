import torch
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification


def fuseModels(weights):
    C_1 = AutoModelForMaskedLM.from_pretrained('parentModel#1')
    C_2 = AutoModelForMaskedLM.from_pretrained('parentModel#2')
    C_3 = AutoModelForSequenceClassification.from_pretrained('parentModel#3', num_labels=10)
    C_4 = AutoModelForSequenceClassification.from_pretrained('parentModel#4', num_labels=5)
    C_Set = (C_1, C_2, C_3, C_4)

    MainStructure = set(C_Set[0].state_dict().keys())
    for C in C_Set:
        MainStructure = MainStructure & C.state_dict().keys()
    order_list = [x for x in C_Set[0].state_dict().keys()]
    MainStructure = sorted(MainStructure, key=lambda x: order_list.index(x))

    fusionModel = AutoModelForSequenceClassification.from_pretrained('parentModel#0', num_labels=2)
    fusionParameters = fusionModel.state_dict()
    for part in MainStructure:
        fusionParameters[part] = torch.zeros_like(fusionParameters[part])
        for i, C in enumerate(C_Set):
            fusionParameters[part] += weights[i] * C.state_dict()[part]
    torch.save(fusionParameters, 'fusionModel.pt')
    fusionModel.load_state_dict(torch.load('fusionModel.pt'))
    fusionModel.save_pretrained('./fusionModel/')
    return 'fusionModel'
