import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def check(pred, labels):
    pred = torch.argmax(pred, dim=1)
    labels = labels.to(device=pred.device)
    mask = pred == labels
    return mask

class DKDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=8.0, temperature=4, warmup=20):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup = warmup

    def forward(self, logits_student, logits_teacher, target, epoch):
        target = target.to(device=logits_student.device)

        def _get_gt_mask(logits, target):
            target = target.reshape(-1)
            mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
            return mask

        def _get_other_mask(logits, target):
            target = target.reshape(-1)
            mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
            return mask

        def cat_mask(t, mask1, mask2):
            t1 = (t * mask1).sum(dim=1, keepdims=True)
            t2 = (t * mask2).sum(dim=1, keepdims=True)
            rt = torch.cat([t1, t2], dim=1)
            return rt

        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)

        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)

        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)

        tckd_loss = (
                F.kl_div(log_pred_student, pred_teacher, reduction='batchmean')
                * (self.temperature ** 2)
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
                F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean')
                * (self.temperature ** 2)
        )
        warmup_weight = min(epoch / self.warmup, 1.0)
        loss_dkd = warmup_weight * (self.alpha * tckd_loss + self.beta * nckd_loss)
        return loss_dkd

def lr_cosine(epoch, epochs):
    e = epoch
    E = epochs
    lr_ratio = e / E
    gamma = 1 - math.cos((math.pi / 2) * lr_ratio) ** 2
    return gamma

def lr_linear(epoch, epochs, warmup=20):
    gamma = min(epoch / warmup, 1.0)
    return gamma

def calibrate_target_weight(mask, logits_teacher, logits_student, labels, T=1.0, beta=0.8):
    error_mask = ~mask
    teacher_logits_error = logits_teacher[error_mask]
    student_logits_error = logits_student[error_mask]
    error_labels = labels[error_mask]
    num_error = error_labels.shape[0]

    if num_error == 0:
        return torch.empty(0, device=logits_teacher.device), torch.empty(0, device=logits_student.device)
    teacher_probs_error = F.softmax(teacher_logits_error / T, dim=1)
    p_tac_t = teacher_probs_error.gather(dim=1, index=error_labels.unsqueeze(1)).squeeze(1)
    teacher_pred_classes = torch.argmax(teacher_probs_error, dim=1)
    p_mac_t = teacher_probs_error.gather(dim=1, index=teacher_pred_classes.unsqueeze(1)).squeeze(1)
    alpha = beta / (p_mac_t - p_tac_t + 1)
    alpha = torch.clamp(alpha, 0.0, 1.0).unsqueeze(1)
    y_onehot = torch.zeros_like(teacher_probs_error)
    y_onehot.scatter_(1, error_labels.unsqueeze(1), 1.0)
    rectified_teacher_probs = alpha * teacher_probs_error + (1 - alpha) * y_onehot
    return rectified_teacher_probs, student_logits_error