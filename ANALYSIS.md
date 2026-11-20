# Cache Line & False Sharing Mechanics — ANALYSIS

Git handle: `abdidulat17-art` | CPU: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | logical cores: 12 | cache line: 64 B | Python 3.14.8 / NumPy 2.5.3
Seed: `5391446366016483103` | manifest: `cache_lab_manifest.json` (sha256 `a39c74895e1e382a...`, verification_passed = **true**)

All timings are the median of 3 runs; 4 threads x 60,000,000 updates each. "ns/update" =
wall-clock elapsed / iterations per thread (each thread's average cost of one update while all 4 run concurrently).

## Q1 — MESI state transitions (stride = 1)

`buffer[0]` and `buffer[1]` are 8 bytes apart, so they live in the same 64-byte line L.

1. Start: L is Shared (S) or Exclusive (E) in Core 0 (the loop loads before it stores); Core 1 holds it Invalid (I) or S.
2. Core 0 stores `buffer[0]`: it needs ownership, so it issues an RFO / invalidate. Core 1's copy goes **S/E -> I**; Core 0's copy goes **-> Modified (M)**.
3. Core 1 stores `buffer[1]` right after: its copy is I, so it issues an RFO. Core 0 snoops it while holding L in M, must supply the dirty data (write-back / cache-to-cache transfer), and goes **M -> I**. Core 1 receives L and ends **I -> M** (via E-for-write).
4. Core 0's next store finds L in I and repeats step 2 with the roles reversed. With 4 threads the line circulates M -> I -> M among 4 cores on every store ("ping-pong"), even though no two threads ever touch the same address.

Why Core 1's store buffer stalls: a store retires into the store buffer but can only *commit* to L1 once the line is held in M/E. x86 (TSO) and ARM commit stores to L1 in program order for visibility, so the RFO miss at the head of the buffer blocks every younger store behind it. A cross-core RFO costs on the order of tens of ns, the buffer fills, and the core stalls on the next store/allocate. In this loop every iteration is a store to a line another core just stole, so the stall repeats on every iteration.

## Q2 — Quantitative slowdown

| Layout | elapsed (ns) | ns / update (per thread) | all 3 runs (ns) |
|---|---:|---:|---|
| stride = 1 (8 B) | 762,423,900 | 12.7071 | 762,423,900, 763,085,300, 747,299,300 |
| stride = 4 (32 B) | 568,087,200 | 9.4681 | 560,570,300, 575,862,000, 568,087,200 |
| stride = 8 (64 B) | 363,550,600 | 6.0592 | 371,786,000, 363,156,300, 363,550,600 |
| stride = 16 (128 B) | 356,947,800 | 5.9491 | 356,477,600, 356,947,800, 360,021,000 |
| register-privatized | 60,828,100 | 1.0138 | 61,457,400, 60,828,100, 60,423,800 |

- Slowdown of stride = 1 relative to stride = 8: **2.097 x** (762,423,900 ns / 363,550,600 ns).
- stride = 4 vs stride = 8: 1.563 x. Privatized baseline vs stride = 8: 5.977 x (stride = 8 / privatized).
- Context switches (stride = 1 / stride = 8): voluntary None / None, involuntary None / None — the slowdown is not scheduler noise; it is cycles lost inside the memory pipeline.
- Interpretation: stride = 8 and the privatized baseline differ by the cost of a real L1 store-to-load forwarding chain (load -> add -> store -> load of the same address, a loop-carried dependency of a few cycles) versus a register add; the remaining gap from stride = 1 to stride = 8 is the pure cross-core coherence latency.

## Q3 — L2 spatial prefetcher and 128-byte pairing

Measured: stride = 8 -> 363,550,600 ns, stride = 16 -> 356,947,800 ns, ratio stride8/stride16 = **1.018**. The two are within ~8%, i.e. no measurable adjacent-line effect on this host: either the spatial prefetcher is off/absent here, or its effect is below run-to-run noise.

Many Intel cores pair two 64-byte lines into an aligned 128-byte sector: when a line is fetched, the L2 "adjacent cache line" (spatial) prefetcher also pulls its buddy (address ^ 64). If thread 0's slot is in line 2k and thread 1's slot is in line 2k+1 (64-byte padding), a store from one core can pull the neighbour line toward that core in a shared state and force the other core to re-obtain ownership with another RFO: a *secondary* invalidation. 64-byte padding therefore still suffers when (a) the base address is not 128-aligned so that two threads' slots fall in the same even/odd pair, (b) the prefetcher is enabled in the BIOS (default on most Intel parts), and (c) the stores are frequent enough that prefetch requests race with the owner's stores. 128-byte spacing puts each slot in its own sector, removing that interaction. CPUs without adjacent-line pairing (many ARM cores, AMD in some configs) show stride 8 ~ stride 16.

## Q4 — Register promotion vs. real memory stores

An optimizing compiler runs loop-invariant code motion / scalar promotion (LICM `promoteLoopAccesses`, mem2reg-style): if the address `buffer[slot]` is loop-invariant and nothing in the loop may alias it, LLVM can load it once before the loop, keep the accumulator in a register, and store it once after the loop (or even fold the loop to `iters * delta`). That is exactly what the privatized kernel does by design, and would turn the experiment into a register benchmark with no coherence traffic.

My kernel prevents it: the address is `j = slot + (buffer[slot] >> 63)`, so each iteration's load/store address depends on the value loaded from memory in the same iteration. LLVM cannot prove `buffer[slot] >> 63 == 0` (it holds only because the counter stays < 2^63), so it cannot prove the address is invariant and must keep a load and a store on every iteration. Evidence from my run (`worker_strided_accumulate.inspect_llvm()`, native kernel function, excluding the CPython wrapper):

- kernel symbol: `noundef i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutab`
- `store i64` count: **5**, `load i64` count: **10**, `lshr i64 ..., 63` present: **True**
- Python object fallback (`PyObject`): **False**, NRT calls: 0

Note that LLVM `store` is not `volatile`: the guarantee here comes from the data dependency, not from a volatile qualifier. The hardware still performs one L1 store per iteration, which is what generates the RFO traffic.

## Q5 — CPython GIL vs. hardware cache-line contention

Experiment (pure-Python threads, same stride-1 pattern `lst[slot] += 1`, 1,000,000 updates per thread; GIL enabled: True): 1 thread = 44,095,900 ns, 4 threads = 196,183,600 ns (**4.449 x** longer for 4x the total work, i.e. no parallel speed-up and extra overhead).

With the GIL only one thread executes bytecode at any instant. The threads take turns, so two cores never store to the same line simultaneously and the MESI ping-pong of Q1 cannot occur; false sharing is *masked*. What you see instead is a software cost: lock hand-off between threads (condition variable / futex wake-ups, the 5 ms switch interval, context switches), cache cold-starts when the interpreter migrates between cores, and convoy effects. Total time grows roughly linearly with the thread count (or worse), so more threads make it slower, not faster. That is *lock contention*: threads wait idle for the lock and consume no memory bandwidth.

With `@njit(nogil=True)` the kernel releases the GIL, four cores run native stores truly in parallel, and the cost moves into hardware: threads never wait on a software lock but on RFO responses (Q1), which is why stride = 1 is 2.097 x slower than stride = 8 in my run while the privatized kernel (parallelism ratio 3.95, i.e. sum of per-thread time / wall time) scales almost perfectly. GIL contention = threads blocked by software; false sharing = threads running but stalled in the memory system.
