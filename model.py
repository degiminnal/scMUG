from utils import *
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


muAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)
thetaAct = lambda x: torch.clamp(torch.nn.functional.softplus(x), 1e-4, 1e4)


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + float('inf'), x)


def _nelem(x):
    nelem = torch.sum(torch.isnan(x).float())
    return torch.where(nelem == 0, torch.tensor(1.0, dtype=x.dtype), nelem)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.sum(x) / nelem


# The NB loss function reference: https://github.com/wangyh082/scBGEDA/blob/main/code/loss.py
def NB(theta, y_pred, y_true, mask=False, debug=True, mean=False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = y_true.float()
    y_pred = y_pred.float() * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = torch.minimum(theta, torch.tensor(1e6))
    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + (
            y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))
    if debug:
        assert torch.isfinite(y_pred).all(), 'y_pred has inf/nans'
        assert torch.isfinite(t1).all(), 't1 has inf/nans'
        assert torch.isfinite(t2).all(), 't2 has inf/nans'
        final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = torch.sum(final) / nelem
        else:
            final = torch.mean(final)
    return final

# The ZINB loss function reference: https://github.com/wangyh082/scBGEDA/blob/main/code/loss.py
def ZINB(pi, theta, y_pred, y_true, ridge_lambda=1.0, mean=True, mask=False, debug=False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_pred, y_true, mean=False, debug=debug) - torch.log(1.0 - pi + eps)
    y_true = y_true.float()
    y_pred = y_pred.float() * scale_factor
    theta = torch.minimum(theta, torch.tensor(1e6))

    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

    result = torch.where(y_true < 1e-8, zero_case, nb_case)
    ridge = ridge_lambda * torch.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = torch.mean(result)

    result = _nan2inf(result)
    return result


class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, input_mu, input_theta, input_pi, target):
        return ZINB(input_pi, input_theta, input_mu, target)


class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, input_mu, input_theta, target):
        return NB(input_theta, input_mu, target)


class GaussianNoise(nn.Module):
    def __init__(self, std=0.15):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        in_dim = dims[0]
        for dim in dims[1:]:
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(GaussianNoise())
            encoder_layers.append(nn.Linear(in_dim, dim))
            in_dim = dim
        self.encoder = nn.Sequential(*encoder_layers[1:])

        decoder_layers = []
        dims.reverse()
        for dim in dims[1:-1]:
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(in_dim, dim))
            in_dim = dim
        decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)
        self.mu_layer = nn.Linear(in_dim, dims[-1])
        self.theta_layer = nn.Linear(in_dim, dims[-1])
        self.pi_layer = nn.Linear(in_dim, dims[-1])
        self.loss = []

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        mu = muAct(self.mu_layer(decoded))
        theta = thetaAct(self.theta_layer(decoded))
        pi = torch.sigmoid(self.pi_layer(decoded))
        return mu, theta, pi

    def encode(self, x):
        return self.encoder(x)


def get_data_loader(x, target, batch_size=32):
    class MyDataset(Dataset):
        def __init__(self, _x, _target):
            self.x = torch.tensor(_x.astype(np.float32))
            self.target = torch.tensor(_target.astype(np.float32))

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.target[idx]

    dataset = MyDataset(x, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


latents = []


def train(model, data_loader, epochs=100, criterion=ZINBLoss(), optimizer=None):
    global latents
    latents = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for (inputs, targets) in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mu, theta, pi = model(inputs)
            if type(criterion).__name__ == "ZINBLoss":
                loss = criterion(mu, theta, pi, targets)
            elif type(criterion).__name__ == "NBLoss":
                loss = criterion(mu, theta, targets)
            elif type(criterion).__name__ == "MSELoss":
                loss = criterion(mu, targets)
            else:
                loss = criterion(mu, targets)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.loss.append(epoch_loss)
        if epoch + 1 < epochs:
            mklog(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        else:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
        if (epoch + 1) % 5 == 0:
            latents.append(get_encoded_output(model, data_loader))


def get_encoded_output(model, data_loader):
    model.eval()
    encoded_outputs = []
    with torch.no_grad():
        for (inputs, targets) in data_loader:
            inputs = inputs.to(device)
            encoded = model.encode(inputs)
            encoded_outputs.append(encoded.cpu().numpy())
    return np.vstack(encoded_outputs)
