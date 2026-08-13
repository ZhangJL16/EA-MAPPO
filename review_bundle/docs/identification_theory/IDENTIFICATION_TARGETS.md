# Identification Targets

## Identification Definition

A target $\psi(\mathcal{M})$ is point identified from the deployed logs over model class $\mathfrak{M}$ if

$$
P_{\mathcal{M}_1}^{\mathrm{obs}}=P_{\mathcal{M}_2}^{\mathrm{obs}}
\Longrightarrow
\psi(\mathcal{M}_1)=\psi(\mathcal{M}_2)
$$

for all $\mathcal{M}_1,\mathcal{M}_2\in\mathfrak{M}$.

## Target 1: Action-Level Collision Boundary

For a declared horizon $H_C$, define

$$
r_C(s,a)
=
\Pr_{\mathcal{M}}\!\left(
\text{collision or near-collision within }H_C
\mid S_t=s,\operatorname{do}(A_t=a)
\right).
$$

The boundary is the level set

$$
\partial\mathcal{A}_{\delta_C}(s)
=
\{a:r_C(s,a)=\delta_C\}.
$$

If a neighborhood of $a$ has zero execution propensity and no structural relation transfers information from observed actions, neither $r_C(s,a)$ nor this boundary is point identified.

## Target 2: Action-Conditioned Energy-to-Charger

For the charger policy $\pi_c$ and charger hitting time $T_c$, define

$$
Q_E^{\pi_c}(s,a)
=
\mathbb{E}_{\mathcal{M}}\left[
\sum_{k=t}^{T_c-1}C_k^E
\mid S_t=s,\operatorname{do}(A_t=a),
A_{t+1:T_c-1}\sim\pi_c
\right].
$$

This target is not identified for a never-executed first action without support or a valid model linking it to observed actions. The charger-policy suffix being observed elsewhere does not identify the missing first-step transition.

## Target 3: Task-Continuation Cost or Value

At a pre-commitment history $h_t$, define

$$
V_{\mathrm{cont}}(h_t)
=
\mathbb{E}_{\mathcal{M}}\left[
\sum_{k=t}^{H}R_k^{\mathrm{task}}
\mid \mathcal{H}_t=h_t,
\text{continue task at }t
\right].
$$

For histories where commitment is deterministic, the continuation action has zero support. The target is not identified without extrapolation, randomized continuation, an instrument, a simulator, or another justified bridge.

## Target 4: Joint Continuation Feasibility

Define

$$
\psi_J(h_t,a)
=
\mathbf{1}\left\{
r_C(h_t,a)\le\delta_C,
\quad
Q_E^{\pi_c}(h_t,a)+m\le E_t^-
\right\}.
$$

This binary target inherits non-identifiability from either component. Conjunction does not create information.

## Target 5: Value of a Policy Outside Filter Support

For target policy $\pi$, define $V^\pi$ under the full environment. If its occupancy $d^\pi(h,a)>0$ where the deployed behavior/filter occupancy $d^\mu(h,a)=0$, standard OPE point identification fails without additional structure.

## Identified, Partially Identified, and Unidentified Regions

Let

$$
\mathcal{I}_{\mathrm{obs}}
=
\{(h,a):d^\mu(h,a)>0\}
$$

be the observed-support region. Then:

- targets on $\mathcal{I}_{\mathrm{obs}}$ may be identified under sequential consistency and no hidden confounding;
- targets outside $\mathcal{I}_{\mathrm{obs}}$ are not point identified nonparametrically;
- smoothness, shape constraints, or model classes can yield partial-identification bounds;
- simulators, instruments, shadow variables, trusted baselines, or randomized probes may restore identification only under their stated validity assumptions.

## Deliberate Non-Assumption

The theory does not assume that every target above is identifiable. The first task is to identify the equivalence class of environments consistent with $\mathcal{D}_{\mathrm{obs}}$ and report the induced identified set.
