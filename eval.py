import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm
from geopy.distance import geodesic as GD


def most_probable_frequent_location(prob_row, gps_gallery, top_k=10, threshold_km=5):
    topk = torch.topk(prob_row, top_k)
    indices = topk.indices.cpu().numpy()
    probs = topk.values.cpu().numpy()

    coords = [tuple(gps_gallery[i].cpu().numpy()) for i in indices]

    # Cluster coordinates based on distance
    groups = []
    for idx, coord in enumerate(coords):
        added = False
        for group in groups:
            if GD(coord, group[0][0]).km < threshold_km:
                group.append((coord, probs[idx], indices[idx]))
                added = True
                break
        if not added:
            groups.append([(coord, probs[idx], indices[idx])])

    # Choose the most frequent group
    most_frequent = max(groups, key=len)

    # Within that group, pick the coordinate with the highest probability
    best = max(most_frequent, key=lambda x: x[1])  # x = (coord, prob, index)
    return best[2]  # return index in gps_gallery


def tensor_to_filepath(tensor):
    return ''.join([chr(int(c)) for c in tensor if c != 0])


def distance_accuracy(targets, preds, dis=2500, gps_gallery=None):
    total = len(targets)
    correct = 0
    gd_avg = 0

    for i in range(total):
        gd = GD(gps_gallery[preds[i]], targets[i]).km
        gd_avg += gd
        if gd <= dis:
            correct += 1

    gd_avg /= total
    return correct / total, gd_avg


def single_distance_accuracy(target, pred, dis=2500, gps_gallery=None):
    correct = 0

    gd = GD(pred, target).km
    if gd <= dis:
        correct = 1

    return correct


'''def eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery.to(device)


    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            # Get predictions (probabilities for each location based on similarity)
            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)

            # Predict gps location with the highest probability (index)
            # outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()

            # My implementation
            outs = []
            for prob_row in probs:
                out = most_probable_frequent_location(prob_row, gps_gallery)
                outs.append(out)

            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1]  # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        accuracy_results[f'acc_{dis}_km'] = acc

    return accuracy_results
'''

def eval_images(val_dataloader, model, gps_gallery, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = gps_gallery.to(device)

    with torch.no_grad():
        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)

            outs = []
            for prob_row in probs:
                out = most_probable_frequent_location(prob_row, gps_gallery)
                outs.append(out)

            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1]  # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        accuracy_results[f'acc_{dis}_km'] = acc

    return accuracy_results

'''def new_eval_images(val_dataloader, model, device="cpu"):
    model.eval()
    preds = []
    targets = []

    gps_gallery = model.gps_gallery

    with torch.no_grad():       
        for img_tensors, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = []

            for img_tensor in img_tensors:
                img_path = tensor_to_filepath(img_tensor)
                img = Image.open(img_path).convert('RGB')
                img = transforms.ToTensor()(img).unsqueeze(0).to(device)
                imgs.append(img)

            imgs = torch.cat(imgs, dim=0)

            logits_per_image = model(imgs, gps_gallery)
            probs = logits_per_image.softmax(dim=-1)

            outs = torch.argmax(probs, dim=-1).detach().cpu().numpy()

            preds.append(outs)
            targets.append(labels)

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    model.train()

    distance_thresholds = [2500, 750, 200, 25, 1]  # km
    accuracy_results = {}
    for dis in distance_thresholds:
        acc, avg_distance_error = distance_accuracy(targets, preds, dis, gps_gallery)
        print(f"Accuracy at {dis} km: {acc}, Average Distance Error: {avg_distance_error}")
        accuracy_results[f'acc_{dis}_km'] = acc

    return accuracy_results'''