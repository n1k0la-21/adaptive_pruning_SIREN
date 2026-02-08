import src.model.SIREN as si
import src.loss.SDF_loss as loss_module
import torch
import numpy as np



def sample_surface(data: np.array, num: int, rng: np.random.Generator):
    return rng.choice(data, size=num, replace=False)

def sample_off_surface(num: int, rng: np.random.Generator):
    return rng.uniform(-1, 1, size=(num, 3))

def train(epochs: int, data: np.array, no_surface: int, no_off_surface:int, model: si.SIRENSDF, loss: loss_module.Loss, optimizer: torch.optim.Adam):
    rng = np.random.default_rng(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for step in range(epochs):
        x_surface = sample_surface(data=data, num=no_surface, rng=rng)
        x_off_surface = sample_off_surface(num=no_off_surface, rng=rng)

        x_surface = torch.from_numpy(x_surface).float().to(device)
        x_off_surface = torch.from_numpy(x_off_surface).float().to(device)

        x_all = torch.cat([x_surface, x_off_surface], dim=0)
        x_all.requires_grad_(True)

        sdf_all = model.forward(x_all)
        sdf_surface = sdf_all[:no_surface]

        grad_all = torch.autograd.grad(outputs=sdf_all, inputs=x_all, grad_outputs=torch.ones_like(sdf_all), create_graph=True)[0]
        grad_surface = grad_all[:no_surface]

        current_loss = loss.compute_loss(sdf_pred=sdf_surface, grad_all=grad_all, grad_surface=grad_surface, normals=None)

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:05d} | Loss {current_loss.item():.6f}")


        