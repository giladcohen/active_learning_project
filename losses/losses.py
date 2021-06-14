import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_loss(s_logits, t_logits):
    return F.kl_div(F.log_softmax(t_logits, dim=1), F.softmax(s_logits, dim=1), reduction="batchmean")

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class VAT(nn.Module):
    def __init__(self, model, n_power, xi, radius):
        super(VAT, self).__init__()
        self.n_power = n_power
        self.XI = xi
        self.model = model
        self.epsilon = radius

    def forward(self, X, logit):
        vat_loss = self.virtual_adversarial_loss(X, logit)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit):
        d = torch.randn_like(x, device='cuda')

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m = self.model(x + d)['logits']
            dist = kl_loss(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        logit_p = logit.detach()
        logit_m = self.model(x + r_vadv)['logits']
        loss = kl_loss(logit_p, logit_m)
        return loss

    def virtual_adversarial_images(self, x, logit):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit)
        return x + r_vadv

