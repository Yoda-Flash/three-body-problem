import numpy as np
import matplotlib.pyplot as plt
import asyncio
import websockets

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

def rk4(a: np.ndarray, system: System, dt: float, num_stages: int, coeff: np.array, weights: np.array) -> None:
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

def rkf4(a: np.ndarray, system: System, dt: float,
         coeff: np.array, weights: np.array, weights_test: np.array, min_power: int, num_stages: int,
         tolerance: double, safety_fac_max: float, safety_fac_min: float, safety_fac: double) -> None:
    compute_accel(a, system)
    rk[0] = system.v
    vk[0] = a

    for stage in range(1, num_stages):
        for i in range(stage):
            temp_system.r[:] += coeff[stage - 1] * rk[i]
            temp_system.v[:] += coeff[stage - 1] * vk[i]

        temp_system.r[:] = system.r + dt * temp_system.r
        temp_system.v[:] = system.v + dt * temp_system.v

        rk[stage] = temp_system.v
        compute_accel(vk[stage], temp_system)

    r_1[:] = system.r
    v_1[:] = system.v
    for stage in range(num_stages):
        r_1[:] += dt * weights[stage] * rk[stage]
        v_1[:] += dt * weights[stage] * vk[stage]
        error_estimation_delta_x[:] += dt * (weights[stage] - weights_test[stage]) * rk[stage]
        error_estimation_delta_v[:] += dt * (weights[stage] - weights_test[stage]) * vk[stage]

    tolerance_scale_x[:] = tolerance + np.maximum(np.abs(system.r), np.abs(r_1)) * tolerance
    tolerance_scale_v[:] = tolerance + np.maximum(np.abs(system.v), np.abs(v_1)) * tolerance
    total = np.average(
        np.square(error_estimation_delta_r / tolerance_scale_r)
    ) + np.average(np.square(error_estimation_delta_v / tolerance_scale_v))
    error = math.sqrt(total / 2.0)

    if error <= 1.0:
        system.r[:] = r_1
        system.v[:] = v_1

    if error < 1e-12:
        error = 1e-12

    dt_new = dt * safety_fac / math.pow(error, 1.0 / (1.0 + float(min_power)))
    if dt_new > safety_fac_max * dt:
        dt *= safety_fac_max
    elif dt_new < safety_fac_min * dt:
        dt *= safety_fac_min
    else:
        dt = dt_new

connected_clients = set()

async def handle_client(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            buf = message
    except websockets.exceptions.ConnectionClosedOK:
        pass

async def main():
    rk4_num_stages = 4
    rk4_coeff = np.array([0.5, 0.5, 1.0])
    rk4_weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0

    rkf4_coeff = np.array((
        [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0],
    ))
    rkf4_weights = np.array(
        [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0]
    )
    rkf4_weights_test = np.array(
        [
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        ]
    )
    rkf4_min_power = 4
    rkf4_num_stages = len(weights)

    safety_fac_max = 6.0
    safety_fac_min = 0.33
    safety_fac = math.pow(0.38, 1.0 / (1.0 + float(min_power)))
    await websockets.serve(handle_client, "three-body-problem.yoda-flash.hackclub.app", 37248)

asyncio.run(main())

if __name__ == "main":
    main()