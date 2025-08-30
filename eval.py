import os
import torch
import json
import numpy as np
from scipy.stats import false_discovery_control
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


def eval_images_detailed(image_dataloader, model):
    model.eval()
    preds = {}
    targets = []

    with torch.no_grad():
        for imgs, labels in tqdm(image_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            for img, label in zip(imgs, labels):
                top_pred_gps, top_pred_prob = model.predict(img, top_k=10)
                preds[img] = top_pred_gps
                targets.append(label)

    # distance_thresholds = [2500, 750, 200, 25, 1]  # km
    distance_thresholds = [2000, 1000, 500, 100, 25, 1]  # km

    accuracy_results = {}
    out_of_thresholds = {str(dis): [] for dis in distance_thresholds}

    for dis in distance_thresholds:
        correct = 0
        for (filename, pred_list), target in zip(preds.items(), targets):
            is_in = False
            out_pred = (0.0, 0.0)
            for pred in pred_list:
                gd = GD(pred, target).km

                # In threshold
                if gd <= dis:
                    is_in = True
                    break
                else:
                    out_pred = pred

            if is_in:
                correct += 1
            else:
                out_of_thresholds[str(dis)].append({
                    "filename": filename,
                    "pred_gps": [float(out_pred[0]), float(out_pred[1])]
                })

        acc = correct / len(targets)
        accuracy_results[f'acc_{dis}_km'] = acc
        print(f"Accuracy at {dis} km: {acc}")

    # Save out-of-threshold filenames to a log file
    log_path = os.path.join(os.getcwd(), "out_of_thresholds.json")
    with open(log_path, 'w') as f:
        json.dump(out_of_thresholds, f, indent=2)

    return accuracy_results


def eval_images(image_dataloader, model):
    model.eval()
    preds = []
    targets = []
    filenames = []

    with torch.no_grad():
        for imgs, labels in tqdm(image_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            for img, label in zip(imgs, labels):
                top_pred_gps, top_pred_prob = model.predict(img, top_k=10)
                preds.append(top_pred_gps[0])
                targets.append(label)
                filenames.append(img)

    # distance_thresholds = [2500, 750, 200, 25, 1]  # km
    distance_thresholds = [2000, 1000, 500, 100, 25, 1] # km

    accuracy_results = {}
    out_of_thresholds = {str(dis): [] for dis in distance_thresholds}

    for dis in distance_thresholds:
        correct = 0
        for pred, target, filename in zip(preds, targets, filenames):
            gd = GD(pred, target).km

            # Out of threshold
            if gd > dis:
                out_of_thresholds[str(dis)].append({
                    "filename": filename,
                    "pred_gps": [float(pred[0]), float(pred[1])]
                })

            # In threshold
            if gd <= dis:
                correct += 1

        acc = correct / len(targets)
        accuracy_results[f'acc_{dis}_km'] = acc
        print(f"Accuracy at {dis} km: {acc}")

    # Save out-of-threshold filenames to a log file
    log_path = os.path.join(os.getcwd(), "out_of_thresholds.json")
    with open(log_path, 'w') as f:
        json.dump(out_of_thresholds, f, indent=2)

    return accuracy_results


def eval_images_weighted(image_dataloader, model):
    model.eval()
    results = []

    with torch.no_grad():
        for imgs, labels in tqdm(image_dataloader, desc="Evaluating"):
            for img, label in zip(imgs, labels):
                top_k_preds_gps, top_k_preds_prob = model.predict(img, top_k=10)

                # Calculate weighted average distance
                weighted_dist_sum = 0
                total_prob = sum(top_k_preds_prob)

                for pred_gps, pred_prob in zip(top_k_preds_gps, top_k_preds_prob):
                    distance_km = GD(pred_gps, label.cpu().numpy()).km
                    weighted_dist_sum += distance_km * (pred_prob / total_prob)

                results.append({
                    "filename": img,
                    "weighted_distance": float(weighted_dist_sum),
                    "target_gps": label.cpu().numpy().tolist(),
                    "top_pred_gps": [float(pred) for pred in top_k_preds_gps[0]],
                    "top_pred_prob": float(top_k_preds_prob[0])
                })

    distance_thresholds = [2000, 1000, 500, 100, 25, 1]  # km
    accuracy_results = {}
    out_of_thresholds = {str(dis): [] for dis in distance_thresholds}

    for dis in distance_thresholds:
        correct = 0
        for res in results:
            if res["weighted_distance"] <= dis:
                correct += 1
            else:
                out_of_thresholds[str(dis)].append({
                    "filename": res["filename"],
                    "weighted_distance": res["weighted_distance"]
                })

        acc = correct / len(results)
        accuracy_results[f'acc_{dis}_km'] = acc
        print(f"Accuracy at {dis} km (weighted): {acc}")

    # Save out-of-threshold filenames to a log file
    log_path = os.path.join(os.getcwd(), "out_of_thresholds_weighted.json")
    with open(log_path, 'w') as f:
        json.dump(out_of_thresholds, f, indent=2)

    return accuracy_results