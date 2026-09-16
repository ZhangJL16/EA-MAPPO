# Finite-Tail Certifiability Kill Gate

## Target

Determine whether the following fixed-policy toy problem produces a nontrivial
theoretical foundation for a new persistent-control paper.

Given \(n\) independent administratively right-censored return times

\[
  X_i=\min\{\tau_i,H\},
  \qquad
  C_i=\mathbf 1\{\tau_i\le H\},
\]

can finite data certify \(\mathbb E[\tau]\le B\), or distinguish

\[
  \mathsf H_0:\mathbb E[\tau]\le B
  \qquad\text{from}\qquad
  \mathsf H_1:\mathbb E[\tau]\ge B+\Delta,
\]

under the weakest physically plausible exponential tail envelope

\[
  \mathcal P(C,\lambda)
  :=
  \left\{
    P:\Pr_P(\tau>t)\le C e^{-\lambda t}
    \quad\forall t\ge0
  \right\}?
\]

The purpose is a kill test, not an algorithm proposal.  POMDPs, policy learning,
history encoders, and UAV dynamics are out of scope.

## Status

**COHERENT AS STATED.** The toy problem admits a clean upper bound, an exact
non-identifiability floor, and a standard parametric sample lower bound.  The
mathematics survives, but the **novelty gate fails**: the result reduces to
classical censored-mean identifiability plus generic concentration/testing
arguments.

## Invariant Object

The invariant object is the uncensored mean return time

\[
  \mu(P):=\mathbb E_P[\tau]
  =\int_0^\infty S_P(t)\,dt,
  \qquad
  S_P(t):=\Pr_P(\tau>t).
\]

The censored data identify only the restricted mean

\[
  m_H(P):=\mathbb E_P[\min\{\tau,H\}]
  =\int_0^H S_P(t)\,dt.
\]

The gap \(r_H(P):=\mu(P)-m_H(P)\) is the unseen-tail contribution.  This
decomposition, rather than a Lyapunov function or learned representation,
organizes every result below.

## Assumptions

1. \(\tau_1,\ldots,\tau_n\) are i.i.d. nonnegative return times under one fixed
   policy and one fixed initial distribution.
2. Censoring is deterministic and administrative at a known horizon \(H>0\).
3. The tail envelope constants \(C\ge1\) and \(\lambda>0\) are known and valid
   uniformly over the candidate class.
4. The envelope is an upper bound only; the conditional distribution beyond
   \(H\) is otherwise unrestricted.
5. The testing gap satisfies \(0<\lambda\Delta\le1/2\) when the exponential
   parametric lower bound is invoked.

These assumptions deliberately favor certifiability.  No claim is made that
\(C\) and \(\lambda\) can be learned from the same censored sample without
further structure.

## Notation

- \(S_P(t)=\Pr_P(\tau>t)\): survival function.
- \(\mu(P)=\mathbb E_P\tau\): unrestricted mean return time.
- \(m_H(P)=\mathbb E_P[\tau\wedge H]\): restricted mean.
- \(r_H(P)=\mu(P)-m_H(P)\): unobserved tail remainder.
- \(R_H:=Ce^{-\lambda H}/\lambda\): uniform upper bound on \(r_H(P)\).
- \(\bar X_n=n^{-1}\sum_iX_i\): empirical restricted mean.
- \(L_\delta:=\log(2/\delta)\).
- \(Q_\theta^H\): law of \((\tau\wedge H,\mathbf1\{\tau\le H\})\) when
  \(\tau\sim\mathrm{Exp}(\theta)\).

## Derivation Strategy

1. Decompose the target mean into an observed restricted mean and an unobserved
   tail remainder.
2. Control the remainder using the known exponential envelope.
3. Estimate the restricted mean with bounded-variable Bernstein concentration.
4. Construct two distributions with identical censored-data laws and different
   post-\(H\) hazards to obtain an exact identifiability lower bound.
5. Restrict to a censored exponential subfamily and apply a two-point testing
   argument to obtain the sampling lower bound.
6. Compare the resulting statements against survival-analysis identifiability,
   RMST, mean-estimation, and finite-sample stability literature.

## Derivation Map

1. The target \(\mu\) depends on \(m_H\) and \(r_H\) through the exact identity
   \(\mu=m_H+r_H\).
2. The envelope gives the deterministic approximation term
   \(0\le r_H\le R_H\).
3. The data estimate \(m_H\); a variance bound derived from the same tail
   envelope gives a Bernstein confidence radius.
4. Two models with identical law below \(H\) and identical censoring mass but
   different post-\(H\) hazards show that \(R_H\) is an information-theoretic,
   not computational, obstruction.
5. Censored exponential models quantify the remaining sampling cost once
   \(H\) is long enough to make the tail floor smaller than \(\Delta\).
6. The upper and lower bounds match in their leading
   \(\lambda^{-2}\Delta^{-2}\log(1/\delta)\) dependence in the estimable regime,
   but this match is obtained from standard statistical ingredients.

## Main Derivation

### Step 1. Exact observed-tail decomposition

For every nonnegative \(\tau\), Tonelli's theorem gives the identity

\[
  \mu(P)
  =\int_0^\infty S_P(t)\,dt
  =\underbrace{\int_0^H S_P(t)\,dt}_{m_H(P)}
   +
   \underbrace{\int_H^\infty S_P(t)\,dt}_{r_H(P)}.
  \tag{1}
\]

Because \(X=\tau\wedge H\), the first term is exactly

\[
  m_H(P)=\mathbb E_P[X].
  \tag{2}
\]

Equations (1)--(2) are identities.  They do not use the tail envelope.

### Step 2. Uniform tail approximation error

For \(P\in\mathcal P(C,\lambda)\),

\[
\begin{aligned}
  0\le r_H(P)
  &=\int_H^\infty S_P(t)\,dt\\
  &\le\int_H^\infty Ce^{-\lambda t}\,dt
  =\frac{C}{\lambda}e^{-\lambda H}
  =R_H.
\end{aligned}
\tag{3}
\]

Thus the envelope converts an unidentified infinite tail into a deterministic
one-sided error bar.  It does not identify the tail law.

### Step 3. A finite-sample upper certificate

The envelope also gives

\[
  \mathbb E_P[\tau^2]
  =2\int_0^\infty tS_P(t)\,dt
  \le \frac{2C}{\lambda^2}
  =:V_\lambda.
  \tag{4}
\]

Therefore \(\operatorname{Var}(X)\le V_\lambda\), while \(0\le X\le H\).
Bernstein's inequality implies that, with probability at least \(1-\delta\),

\[
  |\bar X_n-m_H(P)|
  \le
  a_n(H,\delta)
  :=
  \sqrt{\frac{2V_\lambda L_\delta}{n}}
  +\frac{2HL_\delta}{3n}.
  \tag{5}
\]

Combining (1), (3), and (5) gives the uniform confidence interval

\[
  \boxed{
  \mu(P)\in
  \left[
    (\bar X_n-a_n)_+,
    \bar X_n+a_n+R_H
  \right]
  }
  \tag{6}
\]

for every \(P\in\mathcal P(C,\lambda)\).  In particular,

\[
  U_n:=\bar X_n+a_n+R_H
  \tag{7}
\]

is a valid \((1-\delta)\) upper certificate for mean return time.

This is a proposition, not an approximation.  It is a direct truncation plus
concentration result; the censoring indicator \(C_i\) is not even needed because
administrative censoring makes \(X_i\) observable and bounded.

### Step 4. Sufficient horizon and sample size for a gap \(\Delta\)

Define the midpoint estimator

\[
  \widehat\mu_n:=\bar X_n+\frac{R_H}{2}.
\]

On the event in (5),

\[
  |\widehat\mu_n-\mu(P)|
  \le a_n+\frac{R_H}{2}.
  \tag{8}
\]

It is sufficient for testing two hypotheses separated by \(\Delta\) that

\[
  R_H\le\frac{\Delta}{2},
  \qquad
  a_n\le\frac{\Delta}{4}.
  \tag{9}
\]

The first condition holds when

\[
  H\ge
  \frac1\lambda
  \log\frac{2C}{\lambda\Delta}.
  \tag{10}
\]

Using (4)--(5), the second holds for a universal constant \(c>0\) when

\[
  n\ge
  c\left(
    \frac{C}{\lambda^2\Delta^2}
    +\frac{H}{\Delta}
  \right)
  \log\frac{2}{\delta}.
  \tag{11}
\]

Hence one valid order-level upper bound is

\[
  n=
  O\!\left[
    \left(
      \frac{C}{\lambda^2\Delta^2}
      +
      \frac{1}{\lambda\Delta}
      \log\frac{C}{\lambda\Delta}
    \right)
    \log\frac1\delta
  \right],
  \tag{12}
\]

after choosing the smallest horizon satisfying (10).

### Step 5. Exact non-identifiability below the tail horizon

Set \(C=1\) and let the pre-\(H\) survival law be

\[
  S(t)=e^{-\lambda t},\qquad 0\le t\le H.
\]

Both candidate models therefore have censoring probability
\(q=e^{-\lambda H}\).  Conditional on survival past \(H\), define:

\[
  P_1:\quad S_1(H+s)=q e^{-\lambda s},
\]

and, for any \(M>\lambda\),

\[
  P_0^{(M)}:\quad S_0^{(M)}(H+s)=q e^{-Ms},
  \qquad s\ge0.
\]

Both distributions belong to \(\mathcal P(1,\lambda)\).  They have exactly the
same distribution of \((X,C)\): their event-time laws agree through \(H\), and
both have the same mass \(q\) censored at \(H\).  Nevertheless,

\[
  \mu(P_1)-\mu(P_0^{(M)})
  =q\left(\frac1\lambda-\frac1M\right)
  \xrightarrow[M\to\infty]{}
  \frac{e^{-\lambda H}}{\lambda}.
  \tag{13}
\]

Consequently, for every

\[
  0<\Delta<\frac{e^{-\lambda H}}{\lambda},
  \tag{14}
\]

one can choose \(M\) so that the two means differ by at least \(\Delta\), while
the censored datasets are identically distributed for every sample size.  Any
test has worst-case error at least \(1/2\).

This lower bound matches the envelope remainder in (3) for \(C=1\).  It proves
that the horizon requirement is not an artifact of estimator (7): before
enough follow-up has accumulated, more trajectories cannot repair missing tail
support.

### Step 6. Sampling lower bound after identifiability

Consider the parametric subclass \(\tau\sim\operatorname{Exp}(\theta)\) with
\(\theta\ge\lambda\), which lies in \(\mathcal P(1,\lambda)\).  The censored
likelihood is

\[
  q_\theta(x,c)
  =\bigl(\theta e^{-\theta x}\bigr)^c
   \bigl(e^{-\theta H}\bigr)^{1-c},
\]

with \(x<H\) when \(c=1\) and \(x=H\) when \(c=0\).  Direct expectation under
\(Q_\theta^H\) gives the identity

\[
  D_{\rm KL}(Q_\theta^H\|Q_{\theta'}^H)
  =
  (1-e^{-\theta H})
  \left[
    \log\frac{\theta}{\theta'}-1+\frac{\theta'}{\theta}
  \right].
  \tag{15}
\]

Choose

\[
  \theta_1=\lambda,
  \qquad
  \theta_0=\frac{\lambda}{1-\lambda\Delta}.
\]

Then \(\mu_1-\mu_0=\Delta\).  With \(x=\lambda\Delta\le1/2\),

\[
  -\log(1-x)-x\le x^2,
\]

and

\[
  1-e^{-\theta_0H}
  \le \min\{1,\theta_0H\}
  \le \min\{1,2\lambda H\}.
\]

Therefore

\[
  D_{\rm KL}(Q_{\theta_0}^H\|Q_{\theta_1}^H)
  \le
  \lambda^2\Delta^2\min\{1,2\lambda H\}.
  \tag{16}
\]

Bretagnolle--Huber applied to \(n\) product observations then implies that any
test with both errors at most \(\delta<1/4\) must satisfy, up to universal
constants,

\[
  \boxed{
  n
  =\Omega\!\left(
    \frac{\log(1/\delta)}
    {\lambda^2\Delta^2\min\{1,\lambda H\}}
  \right).
  }
  \tag{17}
\]

Once (10) holds in the interesting small-\(\Delta\) regime,
\(\lambda H\gtrsim1\), so (17) becomes

\[
  n=\Omega\!\left(
    \frac{log(1/\delta)}{\lambda^2\Delta^2}
  \right),
  \tag{18}
\]

matching the leading term of (11)--(12).

### Step 7. Kill-test comparison

The three requested questions now have precise answers.

| Question | Mathematical answer | Research answer |
|---|---|---|
| Uniform high-probability upper bound? | Yes: (6)--(7) | Direct restricted-mean estimation plus envelope remainder |
| Dependence on \(\lambda,H,\Delta,\delta\) necessary? | Yes: identifiability floor (13)--(14) and sampling lower bound (17) | Standard indistinguishability and two-point testing |
| Beyond generic survival/mean/stability theory? | No evidence | Novelty gate fails |

The result is different in *object* from a Lyapunov PAC bound: it estimates a
first-return mean under administrative censoring instead of verifying a drift or
mean-square stability inequality.  But this distinction alone is insufficient.
The proof uses no controlled-process structure, no partial-observation
identifiability mechanism, and no new statistical technique.

## Remarks and Interpretation

1. **Horizon and sample size are not interchangeable.** Equation (13) shows a
   region where \(n=\infty\) still cannot identify \(\mu\); only increasing
   \(H\) or strengthening the tail model can help.
2. **The envelope is doing the extrapolation.** The finite-data certificate is
   conditional on known \((C,\lambda)\).  Estimating those constants from data
   censored at \(H\) would recreate the same unseen-tail problem.
3. **RMST is the observable quantity.** Without extrapolation structure, the
   honest target is \(m_H=\mathbb E[\tau\wedge H]\), not \(\mathbb E\tau\).
4. **The leading sample rate is ordinary mean estimation.** After the horizon
   is long enough, the \(\lambda^{-2}\Delta^{-2}\) term is the variance scale
   of an exponential-tail random variable.
5. **Partial observation has not entered.** Adding observations, latent state,
   or a policy class now would add difficulty but would not retroactively make
   the toy theorem novel.

## Boundaries and Non-Claims

- The derivation treats i.i.d. cycles under a fixed policy; it does not cover
  dependent return cycles, nonstationary worlds, or history-dependent policies.
- It does not certify positive recurrence of a Markov process from one initial
  distribution.  It certifies a distributional mean-return statement under the
  stated sampling law.
- It does not infer the tail envelope from data.
- It does not handle random/informative censoring, off-policy coverage, or
  policy-uniform guarantees.
- It does not claim that constants in (11) or (17) are minimax sharp.
- It does not claim novelty for the decomposition, certificate, or lower-bound
  technique.
- No learning algorithm, representation, POMDP construction, or UAV experiment
  is authorized by this document.

## Open Risks

1. A sharper literature search may find this exact envelope-and-administrative-
   censoring minimax statement already written explicitly; the current verdict
   already assumes that would weaken, not strengthen, the project.
2. Policy-uniform certification would require complexity and coverage terms,
   but absent a new identification mechanism it risks becoming uniform
   concentration over (6).
3. Partial observation could create a new lower bound only if observational
   equivalence interacts nontrivially with tail extrapolation.  No such result
   has been established.
4. A physically justified known tail envelope may itself be as difficult to
   validate as the target recurrence claim.

## Prior-Work Boundary

- Ding and Nan, *Estimating Mean Survival Time: When is it Possible?*, explain
  that unrestricted mean survival is classically estimable when censoring
  support contains survival support, and otherwise requires additional model
  structure; DOI 10.1111/sjos.12112:
  <https://arxiv.org/abs/1307.8369>.
- Restricted mean survival time is routinely estimated by integrating a
  Kaplan--Meier survival curve up to a prespecified horizon; one accessible
  methodological reference is:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC3922847/>.
- Han et al. study finite-trajectory probabilistic mean-square stability through
  Lyapunov conditions in controlled nonlinear systems, demonstrating that
  "finite-sample stability certification" is already an occupied framing:
  <https://www.sciencedirect.com/science/article/abs/pii/S0005109826001482>.
- Wang et al. give PAC stability learning from outputs for a specific partially
  observed switched-linear model class, showing that partial observation alone
  also does not establish novelty:
  <https://www.sciencedirect.com/science/article/pii/S0005109824001365>.
- Pavse et al. study learning stable policies in unbounded state spaces at ICML
  2024, although their contribution is primarily algorithmic/empirical rather
  than this censored-return estimation problem:
  <https://proceedings.mlr.press/v235/pavse24a.html>.

## Research Decision

\[
\boxed{
  \text{Finite-tail certifiability toy pivot: mathematically valid,}\
  \text{but fails the nontriviality/novelty kill criterion.}
}
\]

Under the approved decision rule, the theoretical pivot should stop here rather
than be inflated with POMDP, policy-class, or UAV machinery.  The surviving
project option is an empirical persistent-autonomy paper centered on observed
stuckness, progress predictability, and long-run completed-task throughput, but
that option is not designed or implemented in this round.
