import numpy as np
import matplotlib.pyplot as plt

class System:
    def __init__(self, num_particles: int, r: np.ndarray, v: np.ndarray, m: np.ndarray, G: float) -> None:
        self.num_particles = num_particles
        self.r = r
        self.v = v
        self.m = m
        self.G = G

    def center_particles(self) -> None:
        M = np.sum(self.m)
        r_cm = np.einsum("i,ij->j", self.m, self.r) / M
        v_cm = np.einsum("i,ij->j", self.m, self.v) / M

        self.r -= r_cm
        self.v -= v_cm

def compute_accel(a: np.ndarray, system: System) -> None:
    # Displacement vector
    r_ij = system.r[:, np.newaxis, :] - system.r[np.newaxis, :, :]
    # Distance
    r_norm = np.linalg.norm(r_ij, axis=2)
    # 1 / r^3
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_r_cubed = 1.0 / (r_norm**3)
    np.fill_diagonal(inv_r_cube, 0.0)

    a[:] = system.G * np.einsum("ijk,ij,i->jk", r_ij, inv_r_cube, system.m)

def euler(a: np.ndarray, system: System, dt: float) -> None:
    compute_accel(a, system)
    system.r += system.v * dt
    system.v += a * dt

def euler_cromer(a: np.ndarray, system: System, dt: float) -> None:
    compute_accel(a, system)
    system.v += a * dt
    system.r += system.v * dt

def rk4(a: np.ndarray, system: System, dt: float) -> None:
    num_stages = 4
    coeff = np.array([0.5, 0.5, 1.0])
    weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0

    r0 = system.r.copy()
    v0 = system.v.copy()
    rk = np.zeros((num_stages, system.num_particles, 3))
    vk = np.zeros((num_stages, system.num_particles, 3))

    compute_accel(a, system)
    rk[0] = v0
    vk[0] = a

    for stage in range(1, num_stages):
        system.r = r0 + dt * coeff[stage - 1] * rk[stage - 1]
        compute_accel(a, system)

        rk[stage] = v0 + dt * coeff[stage] * vk[stage - 1]
        vk[stage] = a

    dr = np.einsum("i,ijk->jk", weights, rk)
    dv = np.einsum("i,ijk->jk", weights, vk)

    system.r = r0 + dt * dr
    system.v = v0 + dt * dv

def leapfrog(a: np.ndarray, system: System, dt: float) -> None:
    compute_accel(a, system)
    system.v += a * 0.5 * dt
    system.r += system.v * dt

    compute_accel(a, system)
    system.v += a * 0.5 * dt
