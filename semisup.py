import torch, torchvision
import copy
import torch.nn.functional as F

class Nop:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs):
        return 0

class FixMatch_Distance:
    def __init__(self, model, weak_augment, strong_augment, marginal_distribution, distance_threshold=1500):
        self.model = model
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment
        self.distance_threshold = distance_threshold

    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs, average_latent_space):
        
        weak_imgs = self.weak_augment(unsup_imgs)
        with torch.no_grad():
            weak_logits, latent_space = self.model(weak_imgs)
            probs = weak_logits.softmax(1)
        
        distances = torch.sqrt((latent_space.unsqueeze(1) - average_latent_space.unsqueeze(0)) ** 2).sum(2)
        min_distances, label_idx = torch.min(distances, 1)
        mask = min_distances <= self.distance_threshold
        strong_pred, _ = self.model(self.strong_augment(unsup_imgs[mask]))

        # Should I add the latent space of the pseudo-labels to the ema of the latent spaces?

        return torch.nn.functional.cross_entropy(strong_pred, label_idx[mask]) if mask.sum() > 0 else 0

class FixMatch_Mcdropout:
    def __init__(self, model, weak_augment, strong_augment, marginal_distribution):
        self.model = model
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment

    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs, confidence=0.85, mc_dropout_passes=30):
        weak_imgs = self.weak_augment(unsup_imgs)
        was_training = self.model.training
        self.model.train()
        with torch.no_grad():
            mc_logits = torch.stack([self.model(weak_imgs)[0] for _ in range(mc_dropout_passes)])
        if not was_training:
            self.model.eval()
        mc_probs = torch.softmax(mc_logits, 2)
        mc_logprobs = F.log_softmax(mc_logits, 2)
        mc_mean_probs = torch.mean(mc_probs, 0)
        # confidence threshold
        max_conf, weak_labels = mc_mean_probs.max(1)
        conf_mask = max_conf >= confidence
        # mutual information
        eps = 1e-10
        H_avg = - (mc_mean_probs * torch.log(mc_mean_probs + eps)).sum(1)
        H_each = - (mc_probs * mc_logprobs).sum(2)
        mean_H = H_each.mean(0)
        mi = H_avg - mean_H
        normalized_mi = mi / torch.log(torch.tensor(mc_logits.size(2), dtype=mi.dtype))
        # selected pseudo-labels confidence threshold + mi 
        mi_mask = normalized_mi < (1 - confidence)
        final_mask = conf_mask & mi_mask
        strong_imgs = self.strong_augment(unsup_imgs[final_mask])
        return F.cross_entropy(self.model(strong_imgs)[0], weak_labels[final_mask]) if final_mask.sum() > 0 else 0

class FixMatch:
    # https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html
    def __init__(self, model, weak_augment, strong_augment, marginal_distribution):
        self.model = model
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment

    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs, confidence=0.95):
        weak_imgs = self.weak_augment(unsup_imgs)
        with torch.no_grad():
            weak_logits, _ = self.model(weak_imgs)
        weak_probs = weak_logits.softmax(1)
        max_probs, weak_labels = weak_probs.max(1)
        ix = max_probs >= confidence
        strong_imgs = self.strong_augment(unsup_imgs[ix])
        strong_pred, _ = self.model(strong_imgs)
        return torch.nn.functional.cross_entropy(strong_pred, weak_labels[ix]) if ix.sum() > 0 else 0

class MixMatch:
    # https://proceedings.neurips.cc/paper_files/paper/2019/hash/1cd138d0499a68f4bb72bee04bbec2d7-Abstract.html
    def __init__(self, model, weak_augment, strong_augment, marginal_distribution):
        self.model = model
        self.augment = weak_augment

    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs, K=8, T=0.5, alpha=0.75):
        # we are using K=8 (instead of K=2 like the paper) to make it more comparable with ReMixMatch
        with torch.no_grad():
            avg_probs = sum(self.model(self.augment(unsup_imgs)).softmax(1) for _ in range(K)) / K
        # sharpen
        labels = (avg_probs**(1/T)) / torch.sum(avg_probs**(1/T), 1, True)
        # mixup
        mixup = torchvision.transforms.v2.MixUp(num_classes=avg_probs.shape[1], alpha=alpha)
        unsup_imgs, labels = mixup(unsup_imgs, labels)
        # loss
        probs = self.model(unsup_imgs).softmax(1)
        return torch.mean((probs - labels)**2)

class ReMixMatch:
    # https://openreview.net/forum?id=HklkeR4KPB
    def __init__(self, model, weak_augment, strong_augment, marginal_distribution):
        self.model = model
        self.weak_augment = weak_augment
        self.strong_augment = strong_augment
        self.p = marginal_distribution
        nclasses = len(marginal_distribution)
        self.p_tilde = torch.tensor([[1/nclasses]*nclasses]*128, device=marginal_distribution.device)

    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs, K=8, T=0.5, alpha=0.75, lmbda_U=1.5, lmbda_Ul=0.5):
        # pseudo-labels
        weak_imgs = self.weak_augment(unsup_imgs)
        with torch.no_grad():
            q = self.model(weak_imgs).softmax(1)
        q = q*self.p/self.p_tilde.mean(0, True)  # distribution alignment
        q /= q.sum(0, True)  # normalize
        q = q**(1/T) / torch.sum(q**(1/T), 0, True)  # sharpening and normalize
        # moving average of the model's predictions
        self.p_tilde = torch.cat((self.p_tilde[1:], q.mean(0, True)))
        # sub-datasets
        X_imgs = self.strong_augment(sup_imgs)
        X_labels = torch.nn.functional.one_hot(sup_labels, q.shape[1])
        U_imgs = torch.cat([self.strong_augment(unsup_imgs) for _ in range(K)] + [weak_imgs])
        U_labels = q.repeat((K+1, 1))
        ix = torch.randperm(len(X_imgs)+len(U_imgs))  # shuffle(W)
        W_imgs = torch.cat((X_imgs, U_imgs))[ix]
        W_labels = torch.cat((X_labels, U_labels))[ix]
        # mixup(X,W)
        beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        lmbd = beta_dist.sample([len(X_imgs)]).to(X_imgs.device)
        X_prime_imgs = lmbd[:, None, None, None]*X_imgs + (1-lmbd[:, None, None, None])*W_imgs[:len(X_imgs)]
        X_prime_labels = lmbd[:, None]*X_labels + (1-lmbd[:, None])*W_labels[:len(X_labels)]
        # mixup(U,W)
        beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        lmbd = beta_dist.sample([len(U_imgs)]).to(X_imgs.device)
        U_prime_imgs = lmbd[:, None, None, None]*U_imgs + (1-lmbd[:, None, None, None])*W_imgs[len(X_imgs):]
        U_prime_labels = lmbd[:, None]*U_labels + (1-lmbd[:, None])*W_labels[len(X_labels):]
        # loss
        # note that in the paper they do not seem to apply any supervised term
        # they also have a rotate loss, which we implement separately
        loss_X_prime = torch.nn.functional.cross_entropy(self.model(X_prime_imgs), X_prime_labels)
        loss_U_prime = torch.nn.functional.cross_entropy(self.model(U_prime_imgs), U_prime_labels)
        loss_Ul_prime = torch.nn.functional.cross_entropy(self.model(U_imgs[:len(unsup_imgs)]), q)
        return loss_X_prime + lmbda_U*loss_U_prime + lmbda_Ul*loss_Ul_prime

class Rotate:
    # rotation term from ReMixMatch and other losses
    # note that you must construct a special branch from the encoder that returns 4 possible values
    # for the rot90
    def __init__(self, model, weak_augment, strong_augment, marginal_distribution):
        self.model = model
        self.strong_augment = strong_augment

    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs, lmbda_r=0.5):
        unsup_imgs = self.strong_augment(unsup_imgs)
        r = torch.randint(4, size=len(unsup_imgs))
        unsup_imgs = torch.stack([torch.rot90(img, r, (1, 2)) for r, img in zip(r, unsup_imgs)])
        return lmbda_r*torch.nn.functional.cross_entropy(self.model(unsup_imgs), r)

class DINO:
    # https://arxiv.org/abs/2104.14294
    # version without multi-crop augmentation
    def __init__(self, model, weak_augment, strong_augment, marginal_distribution):
        self.student = model
        self.teacher = copy.deepcopy(model)
        self.augment = strong_augment
        self.C = 0

    def __call__(self, epoch, sup_imgs, sup_labels, unsup_imgs, tps=0.1, m=0.9, l=0.99):
        tpt = min(epoch/30, 1)*(0.07-0.04) + 0.04
        def H(t, s):
            s = torch.nn.functional.log_softmax(s / tps, 1)
            t = torch.softmax((t-self.C)/tpt, 1)
            return -(t*s).sum(1).mean()
        imgs1, imgs2 = self.augment(unsup_imgs), self.augment(unsup_imgs)
        student_logits1, student_logits2 = self.student(imgs1), self.student(imgs2)
        with torch.no_grad():
            teacher_logits1, teacher_logits2 = self.teacher(imgs1), self.teacher(imgs2)
        loss = H(teacher_logits1, student_logits2)/2 + H(teacher_logits2, student_logits1)/2
        # update teacher
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data = l*param_t.data + (1-l)*param_s.data
        self.C = m*self.C + (1-m)*torch.cat((teacher_logits1, teacher_logits2)).mean(0, True)
        return loss
