# Energy-Barrier H-MAPPO

## Problem Model

For UAV `i`, let normalized energy be

```text
e_i(t) = E_i(t) / E_max
```

and let the per-step normalized energy cost be

```text
c_e = energy_decay_per_step / E_max.
```

For an order option `o`, define the conservative route length

```text
D_i(s, o) =
  dist(x_i, pickup_o)
  + dist(pickup_o, dropoff_o)
  + dist(dropoff_o, charging_station)
```

for an unpicked order. If UAV `i` is already carrying the order, use

```text
D_i(s, o) = dist(x_i, dropoff_o) + dist(dropoff_o, charging_station).
```

With maximum step displacement `v_max * dt`, the normalized energy required to
finish the order and return to charging is

```text
R_i(s, o) = c_e * D_i(s, o) / (v_max * dt).
```

The energy barrier margin is

```text
B_i(s, o) = e_i(t) - R_i(s, o) - rho,
```

where `rho >= 0` is a reserve ratio for modeling path inefficiency, obstacle
detours, low-level tracking error, and one-step decision delay.

The high-level order option is feasible only if

```text
B_i(s, o) >= 0.
```

If the learned high-level policy chooses order while no selected order satisfies
the barrier, the executed option is projected to charge:

```text
Pi_E(order) = charge, if B_i(s, o) < 0.
```

The policy is still trained with PPO, but an auxiliary margin loss can bias the
mode distribution:

```text
L_energy =
  relu(-B_i(s, o)) * P_pi(mode = order | s)
  + beta * relu(B_i(s, o)) * P_pi(mode = charge | s).
```

The first term suppresses unsafe order selection; the second term avoids
unnecessary charging when an order is energy-feasible.

## Theorem

Assume:

1. UAV `i` consumes at most `energy_decay_per_step` per environment step.
2. In one step the UAV travels at most `v_max * dt`.
3. At high-level assignment time, an order option is accepted only when
   `B_i(s, o) >= 0`.
4. Once assigned, the order option is locked until pickup/dropoff completion.
5. The realized low-level path length for the locked option plus return to the
   charging station is at most `D_i(s, o) + Delta`, and the reserve `rho`
   satisfies

```text
rho * E_max >= energy_decay_per_step * Delta / (v_max * dt).
```

Then UAV `i` can complete the accepted order and reach the charging station
with non-negative remaining energy. More strongly, its remaining normalized
energy at the station is at least `rho - energy_decay_per_step * Delta /
(E_max * v_max * dt)`.

## Proof

By Assumption 2, traveling a path of length `L` takes at most

```text
N(L) <= L / (v_max * dt)
```

steps in the conservative model. By Assumption 1, the energy required by a path
of length `L` is upper-bounded by

```text
E_req(L) <= energy_decay_per_step * L / (v_max * dt).
```

At the moment an order is accepted, the barrier condition gives

```text
E_i(t) / E_max >=
  energy_decay_per_step * D_i(s, o) / (E_max * v_max * dt) + rho.
```

Multiplying both sides by `E_max`,

```text
E_i(t) >=
  energy_decay_per_step * D_i(s, o) / (v_max * dt) + rho * E_max.
```

The realized path is at most `D_i(s, o) + Delta`, so

```text
E_req <=
  energy_decay_per_step * (D_i(s, o) + Delta) / (v_max * dt).
```

Therefore the remaining energy after completing the order and reaching the
charging station is at least

```text
E_i(t) - E_req
>= rho * E_max - energy_decay_per_step * Delta / (v_max * dt).
```

By Assumption 5 this quantity is non-negative. Hence the projected high-level
option set prevents accepting an order that cannot be completed with enough
energy to return to charging.

## Structural Role

This is not a replacement for PPO. It is a constrained option projection on top
of H-MAPPO:

```text
learned high-level mode -> energy barrier projection -> executed option
```

The proof applies to any policy class because it constrains the executed option
set, not the parameterization of the policy network. The PPO energy margin loss
then improves learnability by pushing the sampled mode distribution toward the
same feasible set used by the projection.

## Capacity-Aware Charging Queue

The energy barrier can increase the number of UAVs choosing charge. If all
charging options use the same station center as the subgoal, the high-level
option projection creates an artificial target collision. To remove this
structural conflict, define a charging slot map

```text
q_k =
  c + r_d [cos(2 pi k / C), sin(2 pi k / C)],       k < C
  c + r_w [cos(2 pi (k-C)/(N-C)), sin(...)],        k >= C,
```

where `c` is the charging station, `C` is charging capacity, `r_d` is a small
docking radius inside the charging tolerance, and `r_w` is a waiting-ring
radius outside the charging tolerance. The first `C` charging UAVs receive dock
slots and the remaining charging UAVs receive waiting slots.

If the slot distance satisfies

```text
||q_a - q_b|| >= 2 * safe_radius + eta,
```

then distinct queue subgoals do not introduce a target-overlap collision. This
does not prove the learned low-level controller is collision-free along the
entire path, but it removes the unavoidable collision caused by assigning
identical charging subgoals to multiple UAVs.
