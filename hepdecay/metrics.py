import torch

def graph_acc(batch_result):
    outputs = batch_result['outputs'][1]
    labels = batch_result['labels'][1]

    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data)
    result = corrects.detach().item() / len(labels)

    return result

def edge_acc(batch_result):
    outputs = batch_result['outputs'][0]
    labels = batch_result['labels'][0].data

    outputs = torch.reshape(outputs, (outputs.size(0)*outputs.size(1), outputs.size(2)))
    labels = torch.reshape(labels, (labels.size(0)*labels.size(1), labels.size(2)))

    _, preds = torch.max(outputs, 1)
    _, labels = torch.max(labels, 1)

    results = []
    corrects = torch.sum(preds == labels)

    for ii in range(6):
        index = labels == ii
        ipreds = preds[index]
        ilabels = labels[index]

        if len(ilabels) == 0:
            results.append(1.)
        else:
            corrects = torch.sum(ipreds == ilabels)
            results.append(corrects.detach().item() / len(ilabels))

    return results
