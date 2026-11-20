import os
import sys
import json
import math
import re
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
STEPS = 400
DIAG_EVERY = 100
LAMBDA_DAMP = 0.15
LOCAL_NY = GLOBAL_NY // 2  # 256
LOCAL_NX = GLOBAL_NX // 2  # 256
PAD = LOCAL_NY + 2         # 258
DX = 1.0 / (GLOBAL_NX + 1)  # 1/513
DY = 1.0 / (GLOBAL_NY + 1)

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(rank)

    # Shift returns (source, dest): along axis 0 with disp=+1 -> (up, down)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_uuid = cart_comm.bcast(gpu_uuid, root=0)
    instance_seed = derive_student_seed(github_user, gpu_uuid)

    return {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": rank,
        "size": size,
        "row_rank": row_rank,
        "col_rank": col_rank,
        "nbrs": [nbr_up, nbr_down, nbr_left, nbr_right],
        "gpu_uuid": gpu_uuid,
        "github_user": github_user.strip(),
        "instance_seed": instance_seed,
    }


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization (no global array)
# ==============================================================================
def initialize_subdomains(ctx):
    row_rank, col_rank = ctx["row_rank"], ctx["col_rank"]
    nbr_up, nbr_down, nbr_left, nbr_right = ctx["nbrs"]

    # Global interior index of local index i (0..257): row_rank*256 + i  (0 and 513 are physical boundary)
    gi = row_rank * LOCAL_NY + np.arange(PAD)
    gj = col_rank * LOCAL_NX + np.arange(PAD)
    x = (gi / (GLOBAL_NY + 1.0))[:, None] * np.ones((1, PAD))  # axis 0 coordinate
    y = np.ones((PAD, 1)) * (gj / (GLOBAL_NX + 1.0))[None, :]  # axis 1 coordinate

    # Diffusivity kappa in [0.08, 0.24]
    kappa = 0.16 + 0.08 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

    # Seed-derived Gaussian pulses (identical on every rank -> consistent global field)
    rng = np.random.default_rng(ctx["instance_seed"])
    n_pulses = 3
    cx = rng.uniform(0.2, 0.8, n_pulses)
    cy = rng.uniform(0.2, 0.8, n_pulses)
    amp = rng.uniform(0.5, 1.0, n_pulses)
    sigma = 0.07
    u = np.zeros((PAD, PAD), dtype=np.float64)
    for k in range(n_pulses):
        u += amp[k] * np.exp(-((x - cx[k]) ** 2 + (y - cy[k]) ** 2) / (2.0 * sigma ** 2))

    # Localized (weak) source
    src = 1.0e-3 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2.0 * 0.05 ** 2))

    # Dirichlet u = 0 on outer global edges
    if nbr_up == MPI.PROC_NULL:
        u[0, :] = 0.0
    if nbr_down == MPI.PROC_NULL:
        u[PAD - 1, :] = 0.0
    if nbr_left == MPI.PROC_NULL:
        u[:, 0] = 0.0
    if nbr_right == MPI.PROC_NULL:
        u[:, PAD - 1] = 0.0

    return (np.ascontiguousarray(u), np.ascontiguousarray(kappa),
            np.ascontiguousarray(src), float(amp.max()))


# ==============================================================================
# TASK 3: Halo pack / unpack kernels and pinned async MPI exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, s_up, s_down, s_left, s_right, ny, nx):
    k = cuda.grid(1)
    if k < nx:
        s_up[k] = u[1, k + 1]        # row 1
        s_down[k] = u[ny, k + 1]     # row ny
    if k < ny:
        s_left[k] = u[k + 1, 1]      # column 1 (strided)
        s_right[k] = u[k + 1, nx]    # column nx (strided)


@cuda.jit
def unpack_halos_kernel(u, r_up, r_down, r_left, r_right, ny, nx):
    k = cuda.grid(1)
    if k < nx:
        u[0, k + 1] = r_up[k]
        u[ny + 1, k + 1] = r_down[k]
    if k < ny:
        u[k + 1, 0] = r_left[k]
        u[k + 1, nx + 1] = r_right[k]


def make_state(ctx, u0, kappa, src):
    st = {}
    st["d_u"] = cuda.to_device(u0)
    st["d_unew"] = cuda.to_device(u0)
    st["d_kappa"] = cuda.to_device(kappa)
    st["d_src"] = cuda.to_device(src)
    lens = [LOCAL_NX, LOCAL_NX, LOCAL_NY, LOCAL_NY]
    st["d_send"] = [cuda.device_array(n, dtype=np.float64) for n in lens]
    st["d_recv"] = [cuda.to_device(np.zeros(n, dtype=np.float64)) for n in lens]
    st["h_send"] = [cuda.pinned_array(n, dtype=np.float64) for n in lens]
    st["h_recv"] = [cuda.pinned_array(n, dtype=np.float64) for n in lens]
    for a in st["h_send"] + st["h_recv"]:
        a[:] = 0.0
    st["stream_interior"] = cuda.stream()
    st["stream_boundary"] = cuda.stream()
    return st


def stage_a_pack_and_d2h(st):
    """Stage A: pack borders on stream_boundary, async D2H into pinned buffers, sync."""
    sb = st["stream_boundary"]
    ds = st["d_send"]
    threads = 128
    blocks = (max(LOCAL_NY, LOCAL_NX) + threads - 1) // threads
    pack_halos_kernel[blocks, threads, sb](st["d_u"], ds[UP], ds[DOWN], ds[LEFT], ds[RIGHT],
                                           LOCAL_NY, LOCAL_NX)
    for k in range(4):
        ds[k].copy_to_host(st["h_send"][k], stream=sb)
    sb.synchronize()


def exchange_halos_async(ctx, st):
    """Stage A + post non-blocking MPI. Returns the list of requests."""
    stage_a_pack_and_d2h(st)
    cart = ctx["cart_comm"]
    nu, nd, nl, nr = ctx["nbrs"]
    hs, hr = st["h_send"], st["h_recv"]
    # Tags name the direction of travel: 0 = moving up, 1 = moving down, 2 = moving left, 3 = moving right
    reqs = [
        cart.Irecv([hr[UP], MPI.DOUBLE], source=nu, tag=1),     # ghost row 0 <- data moving down
        cart.Irecv([hr[DOWN], MPI.DOUBLE], source=nd, tag=0),   # ghost row ny+1 <- data moving up
        cart.Irecv([hr[LEFT], MPI.DOUBLE], source=nl, tag=3),   # ghost col 0 <- data moving right
        cart.Irecv([hr[RIGHT], MPI.DOUBLE], source=nr, tag=2),  # ghost col nx+1 <- data moving left
        cart.Isend([hs[UP], MPI.DOUBLE], dest=nu, tag=0),
        cart.Isend([hs[DOWN], MPI.DOUBLE], dest=nd, tag=1),
        cart.Isend([hs[LEFT], MPI.DOUBLE], dest=nl, tag=2),
        cart.Isend([hs[RIGHT], MPI.DOUBLE], dest=nr, tag=3),
    ]
    return reqs


def finish_halo_exchange(ctx, st, reqs):
    """Stage C (first half): Waitall, async H2D, unpack ghost cells on stream_boundary."""
    MPI.Request.Waitall(reqs)
    sb = st["stream_boundary"]
    dr = st["d_recv"]
    for k in range(4):
        dr[k].copy_to_device(st["h_recv"][k], stream=sb)
    threads = 128
    blocks = (max(LOCAL_NY, LOCAL_NX) + threads - 1) // threads
    unpack_halos_kernel[blocks, threads, sb](st["d_u"], dr[UP], dr[DOWN], dr[LEFT], dr[RIGHT],
                                             LOCAL_NY, LOCAL_NX)


# ==============================================================================
# TASK 4: 2D shared-memory tiled stencil + dual-stream orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, u_new, kappa, src, r_start, r_end, c_start, c_end,
                         dt, inv_dx2, inv_dy2, lam):
    smem = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = r_start + cuda.blockIdx.y * TILE_DIM + ty
    j = c_start + cuda.blockIdx.x * TILE_DIM + tx
    H = u.shape[0]
    W = u.shape[1]

    # Load center cell and the 1-cell halo apron (loads depend only on array bounds,
    # so threads outside the bounding box still provide neighbours for in-box threads)
    if i < H and j < W:
        smem[ty + 1, tx + 1] = u[i, j]
        if ty == 0 and i >= 1:
            smem[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1 and i + 1 < H:
            smem[TILE_DIM + 1, tx + 1] = u[i + 1, j]
        if tx == 0 and j >= 1:
            smem[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1 and j + 1 < W:
            smem[ty + 1, TILE_DIM + 1] = u[i, j + 1]

    cuda.syncthreads()

    if i >= r_start and i <= r_end and j >= c_start and j <= c_end:
        uc = smem[ty + 1, tx + 1]
        un = smem[ty, tx + 1]
        us = smem[ty + 2, tx + 1]
        uw = smem[ty + 1, tx]
        ue = smem[ty + 1, tx + 2]
        kc = kappa[i, j]
        kn = 0.5 * (kappa[i - 1, j] + kc)
        ks = 0.5 * (kappa[i + 1, j] + kc)
        kw = 0.5 * (kappa[i, j - 1] + kc)
        ke = 0.5 * (kappa[i, j + 1] + kc)
        rhs = ((ks * (us - uc) - kn * (uc - un)) * inv_dx2
               + (ke * (ue - uc) - kw * (uc - uw)) * inv_dy2
               - lam * uc * uc * uc + src[i, j])
        u_new[i, j] = uc + dt * rhs


def launch_stencil(st, stream, r0, r1, c0, c1, dt):
    gy = (r1 - r0 + 1 + TILE_DIM - 1) // TILE_DIM
    gx = (c1 - c0 + 1 + TILE_DIM - 1) // TILE_DIM
    tiled_stencil_kernel[(gx, gy), (TILE_DIM, TILE_DIM), stream](
        st["d_u"], st["d_unew"], st["d_kappa"], st["d_src"],
        r0, r1, c0, c1, dt, 1.0 / (DX * DX), 1.0 / (DY * DY), LAMBDA_DAMP)


def step_overlapped(ctx, st, dt):
    ny, nx = LOCAL_NY, LOCAL_NX
    si, sb = st["stream_interior"], st["stream_boundary"]
    # Stage A + B: pack/D2H, then post MPI and immediately launch deep-interior stencil
    reqs = exchange_halos_async(ctx, st)
    launch_stencil(st, si, 2, ny - 1, 2, nx - 1, dt)
    # Stage C: wait, H2D, unpack, boundary ring stencil on stream_boundary
    finish_halo_exchange(ctx, st, reqs)
    launch_stencil(st, sb, 1, 1, 1, nx, dt)                # row 1
    launch_stencil(st, sb, ny, ny, 1, nx, dt)              # row ny
    launch_stencil(st, sb, 2, ny - 1, 1, 1, dt)            # column 1
    launch_stencil(st, sb, 2, ny - 1, nx, nx, dt)          # column nx
    sb.synchronize()
    si.synchronize()
    st["d_u"], st["d_unew"] = st["d_unew"], st["d_u"]


def step_serialized(ctx, st, dt):
    ny, nx = LOCAL_NY, LOCAL_NX
    sb = st["stream_boundary"]
    reqs = exchange_halos_async(ctx, st)
    finish_halo_exchange(ctx, st, reqs)
    sb.synchronize()
    launch_stencil(st, sb, 1, ny, 1, nx, dt)               # whole owned block, after halos
    sb.synchronize()
    st["d_u"], st["d_unew"] = st["d_unew"], st["d_u"]


# ==============================================================================
# TASK 5: Reductions, PTX audit, benchmark, manifest
# ==============================================================================
def allreduce_scalar(cart, val, op):
    out = np.zeros(1, dtype=np.float64)
    cart.Allreduce(np.array([val], dtype=np.float64), out, op=op)
    return float(out[0])


def diagnostics(ctx, cur, prev):
    cart = ctx["cart_comm"]
    own = cur[1:LOCAL_NY + 1, 1:LOCAL_NX + 1]
    e_loc = float(own.sum() * DX * DY)
    energy = allreduce_scalar(cart, e_loc, MPI.SUM)
    if prev is None:
        return energy, 0.0
    diff = own - prev[1:LOCAL_NY + 1, 1:LOCAL_NX + 1]
    res = math.sqrt(allreduce_scalar(cart, float((diff * diff).sum()), MPI.SUM))
    return energy, res


def timed_steps(ctx, st, n, step_fn, dt):
    cart = ctx["cart_comm"]
    cuda.synchronize()
    cart.Barrier()
    ev0, ev1 = cuda.event(), cuda.event()
    t0 = MPI.Wtime()
    ev0.record()
    for _ in range(n):
        step_fn(ctx, st, dt)
    ev1.record()
    ev1.synchronize()
    t1 = MPI.Wtime()
    return (t1 - t0) * 1e3, float(cuda.event_elapsed_time(ev0, ev1))


def run_mode(ctx, mode, fields, dt):
    u0, kappa, src = fields
    step_fn = step_overlapped if mode == "overlapped" else step_serialized
    cart = ctx["cart_comm"]

    # Warm-up on a throwaway state (JIT compile, MPI wire-up)
    warm = make_state(ctx, u0, kappa, src)
    for _ in range(3):
        step_fn(ctx, warm, dt)
    cuda.synchronize()
    del warm

    st = make_state(ctx, u0, kappa, src)
    energies, residuals, steps_logged = [], [], [0]
    e0, r0 = diagnostics(ctx, st["d_u"].copy_to_host(), None)
    energies.append(e0)
    residuals.append(r0)
    wall_ms = 0.0
    event_ms = 0.0
    done = 0
    while done < STEPS:
        chunk = min(DIAG_EVERY, STEPS - done)
        w1, c1 = timed_steps(ctx, st, chunk - 1, step_fn, dt)
        cuda.synchronize()
        prev = st["d_u"].copy_to_host()
        w2, c2 = timed_steps(ctx, st, 1, step_fn, dt)
        cur = st["d_u"].copy_to_host()
        wall_ms += w1 + w2
        event_ms += c1 + c2
        done += chunk
        e, r = diagnostics(ctx, cur, prev)
        energies.append(e)
        residuals.append(r)
        steps_logged.append(done)

    wall_max = allreduce_scalar(cart, wall_ms, MPI.MAX)
    event_max = allreduce_scalar(cart, event_ms, MPI.MAX)
    final_u = st["d_u"].copy_to_host()
    return {"mode": mode, "wall_ms": wall_max, "event_ms": event_max,
            "energies": energies, "residuals": residuals,
            "steps": steps_logged, "final_u": final_u}


def ptx_audit():
    asm = tiled_stencil_kernel.inspect_asm()
    text = "\n".join(asm.values()) if isinstance(asm, dict) else str(asm)
    counts = {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", text)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", text)),
        "bar.sync": len(re.findall(r"\bbar\.sync\b", text)),
    }
    return counts, all(v > 0 for v in counts.values())


def run_solver(ctx):
    cart = ctx["cart_comm"]
    rank = ctx["rank"]
    u0, kappa, src, _ = initialize_subdomains(ctx)

    kmax = allreduce_scalar(cart, float(kappa.max()), MPI.MAX)
    dt_max = DX * DX / (4.0 * kmax)
    dt = 0.5 * dt_max

    if rank == 0:
        print("=" * 72)
        print(f"GITHUB_USERNAME   : {ctx['github_user']}")
        print(f"GPU UUID          : {ctx['gpu_uuid']}")
        print(f"INSTANCE_SEED     : {ctx['instance_seed']}")
        print(f"MPI ranks         : {ctx['size']} (2 x 2 Cartesian, non-periodic)")
        print(f"local padded array: {PAD} x {PAD}, global interior {GLOBAL_NY} x {GLOBAL_NX}")
        print(f"dx = dy = 1/513   : {DX:.9e}")
        print(f"kappa_max         : {kmax:.6f}")
        print(f"dt_max (CFL)      : {dt_max:.6e}")
        print(f"dt used           : {dt:.6e}  (= 0.5 * dt_max)")
        print("=" * 72)
    for r in range(4):
        cart.Barrier()
        if r == rank:
            print(f"[rank {rank}] coords=({ctx['row_rank']},{ctx['col_rank']}) "
                  f"up={ctx['nbrs'][0]} down={ctx['nbrs'][1]} "
                  f"left={ctx['nbrs'][2]} right={ctx['nbrs'][3]}", flush=True)
    cart.Barrier()

    ser = run_mode(ctx, "serialized", (u0, kappa, src), dt)
    ovl = run_mode(ctx, "overlapped", (u0, kappa, src), dt)

    own = (slice(1, LOCAL_NY + 1), slice(1, LOCAL_NX + 1))
    loc_diff = float(np.max(np.abs(ser["final_u"][own] - ovl["final_u"][own])))
    parity = allreduce_scalar(cart, loc_diff, MPI.MAX)

    counts, audit_ok = ptx_audit()

    if rank == 0:
        print("\nStep | Energy E (Allreduce SUM)   | L2 residual ||u^{n+1}-u^n||")
        for k, s in enumerate(ovl["steps"]):
            print(f"{s:4d} | {ovl['energies'][k]:.12e} | {ovl['residuals'][k]:.6e}")
        monotone = all(ovl["energies"][k + 1] <= ovl["energies"][k]
                       for k in range(len(ovl["energies"]) - 1))
        print(f"\nEnergy monotonically non-increasing: {monotone}")
        print(f"serialized_ms (wall, max over ranks) : {ser['wall_ms']:.3f}")
        print(f"overlapped_ms (wall, max over ranks) : {ovl['wall_ms']:.3f}")
        print(f"serialized CUDA-event ms             : {ser['event_ms']:.3f}")
        print(f"overlapped CUDA-event ms             : {ovl['event_ms']:.3f}")
        print(f"overlap speedup (wall)               : {ser['wall_ms'] / ovl['wall_ms']:.4f}x")
        print(f"parity_max_diff                      : {parity:.3e}")
        print(f"PTX counts                           : {counts}")
        print(f"ptx_audit_passed                     : {audit_ok}")

        manifest = {
            "GITHUB_USERNAME": ctx["github_user"],
            "gpu_uuid": ctx["gpu_uuid"],
            "instance_seed": ctx["instance_seed"],
            "mpi_ranks": ctx["size"],
            "process_grid": [2, 2],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_padded_shape": [PAD, PAD],
            "steps": STEPS,
            "dx": DX,
            "kappa_max": kmax,
            "dt_max": dt_max,
            "dt": dt,
            "energy_steps": ovl["steps"],
            "energy_trace_overlapped": ovl["energies"],
            "energy_trace_serialized": ser["energies"],
            "residual_trace_overlapped": ovl["residuals"],
            "energy_monotone_non_increasing": monotone,
            "serialized_ms": ser["wall_ms"],
            "overlapped_ms": ovl["wall_ms"],
            "serialized_cuda_event_ms": ser["event_ms"],
            "overlapped_cuda_event_ms": ovl["event_ms"],
            "parity_max_diff": parity,
            "parity_passed": parity < 1e-12,
            "ptx_counts": counts,
            "ptx_audit_passed": audit_ok,
        }
        with open("submission_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print("\nWrote submission_manifest.json")

    cart.Barrier()
    if not (parity < 1e-12 and audit_ok):
        raise RuntimeError("Verification failed (parity or PTX audit).")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: distributed_cuda_solver.py <github_username>")
    ctx = init_distributed_context(sys.argv[1])
    run_solver(ctx)


if __name__ == "__main__":
    main()
