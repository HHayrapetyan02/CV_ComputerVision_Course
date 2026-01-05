import torch
import torch.optim as optim
import numpy as np

from torch import nn
from collections import OrderedDict


# ============================== 1 Classifier model ============================
def get_cls_model():
    """
    :return: nn model for classification
    """
    # your code here \/
    input_shape = (1, 40, 100) # (n_channels, n_rows, n_cols)

    classification_model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)),
        ('bn1', nn.BatchNorm2d(32)),
        ('relu1', nn.ReLU()),
        ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),

        ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
        ('bn2', nn.BatchNorm2d(64)),
        ('relu2', nn.ReLU()),
        ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),

        ('conv3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
        ('bn3', nn.BatchNorm2d(128)),
        ('relu3', nn.ReLU()),
        ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),

        ('conv4', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
        ('bn4', nn.BatchNorm2d(256)),
        ('relu4', nn.ReLU()),

        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(256 * 5 * 12, 512)),
        ('relu5', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(512, 2))
    ]))

    return classification_model
    # your code here /\


def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    N = X.shape[0]
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    ITERS = N // BATCH_SIZE + 1

    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0

        for i in range(ITERS):
            X_batch, y_batch = X[BATCH_SIZE*i:BATCH_SIZE*(i + 1)], y[BATCH_SIZE*i:BATCH_SIZE*(i + 1)]

            optimizer.zero_grad()
            p_batch = model(X_batch)
            loss = criterion(p_batch, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} loss : {running_loss:.3f}")

    # train model
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    conv_fc1 = nn.Conv2d(256, 512, kernel_size=(5, 12), padding="valid")
    conv_fc2 = nn.Conv2d(512, 2, kernel_size=(1, 1), padding="valid")

    with torch.no_grad():
        conv_fc1.weight = torch.nn.Parameter(cls_model.fc1.weight.reshape(512, 256, 5, 12))
        conv_fc2.weight = torch.nn.Parameter(cls_model.fc2.weight.reshape(2, 512, 1, 1))

    detection_model = nn.Sequential(OrderedDict([
        ('conv1', cls_model.conv1),
        ('bn1', cls_model.bn1),
        ('relu1', cls_model.relu1),
        ('pool1', cls_model.pool1),

        ('conv2', cls_model.conv2),
        ('bn2', cls_model.bn2),
        ('relu2', cls_model.relu2),
        ('pool2', cls_model.pool2),

        ('conv3', cls_model.conv3),
        ('bn3', cls_model.bn3),
        ('relu3', cls_model.relu3),
        ('pool3', cls_model.pool3),

        ('conv4', cls_model.conv4),
        ('bn4', cls_model.bn4),
        ('relu4', cls_model.relu4),

        ('conv_fc1', conv_fc1),
        ('relu5', cls_model.relu5),
        ('dropout', cls_model.dropout),
        ('conv_fc2', conv_fc2),
        ('activation', nn.Softmax(dim=1))
    ]))

    return detection_model
    # your code here /\


# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/

    detections = {}
    detection_model.eval()

    for filename, image in dictionary_of_images.items():
        high, width = image.shape
        img_tensor = torch.FloatTensor(image).expand(1, 1, *image.shape)

        detection = detection_model(img_tensor)
        detection = detection[0, 1].detach().numpy()
        
        image_detections = []
        for i in range(high):
            for j in range(width):
                confidence = detection[i, j]
                row, col = i * 8, j * 8
                
                if row + 40 <= high and col + 100 <= width:
                    image_detections.append([row, col, 40, 100, confidence])
        
        detections[filename] = np.array(image_detections)
    
    return detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    row_1, col_1, high_1, width_1 = first_bbox
    row_2, col_2, high_2, width_2 = second_bbox

    inter_h = max(0, min(row_1 + high_1, row_2 + high_2) - max(row_1, row_2))
    inter_w = max(0, min(col_1 + width_1, col_2 + width_2) - max(col_1, col_2))

    intersection = inter_h * inter_w
    union = high_1 * width_1 + high_2 * width_2 - intersection

    return intersection / union

    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/

    TP, FP = {}, {}
    num_bboxes = 0
    
    for filename, detections in pred_bboxes.items():
        detections_sorted = sorted(detections, key=lambda x: x[4], reverse=True)
        gt_list = gt_bboxes[filename].copy()
        num_bboxes += len(gt_list)

        TP[filename], FP[filename] = [], []

        for pred in detections_sorted:
            best_iou, best_index = 0, -1
            for i, gt_bbox in enumerate(gt_list):
                iou = calc_iou(pred[:4], gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_index = i

            if best_iou >= 0.5:
                TP[filename].append(pred[-1])
                del gt_list[best_index]
            else:
                FP[filename].append(pred[-1])

    TP = sum([tp for tp in TP.values()], [])
    FP = sum([fp for fp in FP.values()], [])

    total, tp_sorted = sorted(TP + FP), sorted(TP)                           

    PR_CURVE = []
    i = 0
    last_c = None

    for j, conf in enumerate(total):
        if last_c != conf:
            tp_fp_above_c = len(total) - j
            while (i < len(tp_sorted)) and (tp_sorted[i] < conf):
                i += 1
            tp_above_c = len(tp_sorted) - i
            recall = tp_above_c / num_bboxes
            precision = tp_above_c / tp_fp_above_c
            PR_CURVE.append((recall, precision, conf))
            last_c = conf

    PR_CURVE.append((0, 1, 1))
    PR_CURVE = np.array(PR_CURVE)

    precission, recall = PR_CURVE[:, 1], PR_CURVE[:, 0]
    auc_value = sum(0.5 * (precission[:-1] + precission[1:]) * (recall[:-1] - recall[1:]))

    return auc_value
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.5):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    deetctions_filtered = {}

    for filename, detections in detections_dictionary.items():
        detections_sorted = sorted(detections, key=lambda x: x[4], reverse=True)

        selected = []
        while detections_sorted:
            current = detections_sorted.pop(0)
            selected.append(current)

            keep_idx = []

            for i, detection in enumerate(detections_sorted):
                iou = calc_iou(current[:-1], detection[:-1])
                if iou <= iou_thr:
                    keep_idx.append(i)

            detections_sorted = [detections_sorted[i] for i in keep_idx]

        deetctions_filtered[filename] = np.array(selected)

    return deetctions_filtered                
    # your code here /\
