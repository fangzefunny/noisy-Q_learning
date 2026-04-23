# Estimating Human Sequential Behavioral Likelihood with a Particle Filter

This note explains how to use a particle filter to estimate the likelihood of
human sequential behavioral data. We use the following notation:

- $s_t$: task state or feedback available around trial $t$
- $a_t$: human action on trial $t$
- $m_t$: latent variable on trial $t$
- $\theta$: model parameters

In this repository, the latent variable $m_t$ is represented by Q-values and is
stored as `particle["q"]` in `utils/fit.py`.

## Full Trajectory Likelihood

Suppose we observe one human trajectory

$$
\tau = (s_1, a_1, s_2, a_2, \ldots, s_T, a_T).
$$

The behavioral model assumes that actions are generated from an unobserved
latent process $m_{1:T}$. For example, $m_t$ may represent the subject's internal
value estimates, beliefs, or noisy learning state. The likelihood of the
observed actions is obtained by marginalizing over all possible latent
trajectories:

$$
p(a_{1:T} \mid s_{1:T}, \theta)
= \int p(a_{1:T}, m_{0:T} \mid s_{1:T}, \theta)\,dm_{0:T}.
$$

Equivalently, the full log likelihood is

$$
\log p(a_{1:T} \mid s_{1:T}, \theta)
= \log \int p(m_0 \mid \theta)
      \prod_{t=1}^T
      p(a_t \mid m_t, s_t, a_{t-1}, \theta)
      p(m_t \mid m_{t-1}, s_t, a_{t-1}, \theta)
      \,dm_{0:T}.
$$

This expression is the target quantity for model fitting. If we can evaluate it
for any parameter vector $\theta$, then we can fit $\theta$ by maximizing the log
likelihood, or equivalently by minimizing the negative log likelihood.

## Assumptions and Approximations

The full likelihood above is usually intractable because it integrates over a
high-dimensional latent trajectory. The particle filter becomes useful after
making several standard assumptions.

### 1. Markov latent dynamics

The current latent variable depends on the previous latent variable and the
current trial information, but not directly on the entire past:

$$
p(m_t \mid m_{0:t-1}, s_{1:t}, a_{1:t-1}, \theta)
= p(m_t \mid m_{t-1}, s_t, a_{t-1}, \theta).
$$

In the code, this transition is implemented by:

```python
model.update_latent(particle["q"], r_sample, config, rng)
```

Here `r_sample` is the feedback saved from the previous trial. For the noisy
Q-learning model, this update is:

$$
m_t = m_{t-1} + \alpha (r_{t-1} - m_{t-1}) + \varepsilon_t,
$$

where $\varepsilon_t$ is process noise controlled by $\sigma$.

### 2. Conditional action policy

The observed action depends on the current latent variable and a small set of
behaviorally relevant variables:

$$
p(a_t \mid m_{0:t}, s_{1:t}, a_{1:t-1}, \theta)
= p(a_t \mid m_t, s_t, a_{t-1}, \theta).
$$

In the code, this action likelihood is implemented by:

```python
model.log_policy(particle["q"], prev_a, config)
```

For the noisy Q-learning model, the policy uses the Q-value difference and a
perseveration term:

$$
\operatorname{logit}_t
= \beta (m_{t,B} - m_{t,A}) + \eta\,\operatorname{per}(a_{t-1}),
$$

then converts this logit into probabilities for actions A and B.

### 3. Sequential factorization

The likelihood can be decomposed trial by trial:

$$
p(a_{1:T} \mid s_{1:T}, \theta)
= \prod_{t=1}^T p(a_t \mid a_{1:t-1}, s_{1:t}, \theta).
$$

Therefore,

$$
\log p(a_{1:T} \mid s_{1:T}, \theta)
= \sum_{t=1}^T \log p(a_t \mid a_{1:t-1}, s_{1:t}, \theta).
$$

The challenge is that each one-step predictive likelihood still requires
integrating over the filtering distribution of the latent variable:

$$
p(a_t \mid a_{1:t-1}, s_{1:t}, \theta)
= \int p(a_t \mid m_t, s_t, a_{t-1}, \theta)
       p(m_t \mid a_{1:t-1}, s_{1:t}, \theta)
       \,dm_t.
$$

### 4. Monte Carlo approximation

The particle filter approximates the filtering distribution with $N$ particles:

$$
p(m_t \mid a_{1:t-1}, s_{1:t}, \theta)
\approx \frac{1}{N} \sum_{i=1}^N \delta(m_t - m_t^{(i)}).
$$

Substituting this empirical distribution into the predictive likelihood gives:

$$
p(a_t \mid a_{1:t-1}, s_{1:t}, \theta)
\approx \frac{1}{N} \sum_{i=1}^N
p(a_t \mid m_t^{(i)}, s_t, a_{t-1}, \theta).
$$

The log likelihood contribution of trial `t` is therefore estimated as:

$$
\log p(a_t \mid a_{1:t-1}, s_{1:t}, \theta)
\approx
\log \left( \frac{1}{N} \sum_{i=1}^N w_t^{(i)} \right),
$$

where

$$
w_t^{(i)} = p(a_t \mid m_t^{(i)}, s_t, a_{t-1}, \theta).
$$

In practice, the implementation stores log weights and uses a log-sum-exp trick
for numerical stability:

$$
\log \left( \frac{1}{N} \sum_i \exp(\ell_i) \right)
= b + \log \sum_i \exp(\ell_i - b) - \log N,
$$

where $\ell_i = \log w_t^{(i)}$ and $b = \max_i \ell_i$.

## Pseudocode Matching `utils/fit.py`

The following pseudocode mirrors `estimate_nll_pf` in `utils/fit.py`, but uses
`m` for the latent variable.

```text
function estimate_nll_particle_filter(data, model_class, theta, N, seed):
    model = model_class()
    config = model.load_configs(theta)
    rng = random_generator(seed)

    particles = []
    for i in 1:N:
        m_i = model._init_model(config)
        particles.append(m_i)

    log_likelihood = 0
    previous_action = none
    previous_feedback = none

    for trial t in sorted(data):
        s_t = data[t]
        a_t = observed action from s_t

        for particle i in 1:N:
            if t is not the first trial:
                particles[i] = model.update_latent(
                    particles[i],
                    previous_feedback,
                    config,
                    rng
                )

            log_policy = model.log_policy(
                particles[i],
                previous_action,
                config
            )
            log_weight[i] = log_policy[a_t]

        b = max(log_weight)
        unnormalized_weight[i] = exp(log_weight[i] - b), for all i
        weight_sum = sum_i unnormalized_weight[i]

        log_likelihood += b + log(weight_sum) - log(N)

        normalized_weight[i] = unnormalized_weight[i] / weight_sum

        previous_action = a_t
        previous_feedback = feedback stored in s_t

        ancestor_index = systematic_resampling(normalized_weight, rng)
        particles = copy particles selected by ancestor_index

    return -log_likelihood
```

The systematic resampling step is:

```text
function systematic_resampling(weights, rng):
    N = number of particles
    cumulative_weight = cumulative_sum(weights)
    cumulative_weight[N - 1] = 1

    u_0 = uniform(0, 1 / N)
    proposal[i] = u_0 + i / N, for i = 0, ..., N - 1

    i = 0
    j = 0
    while i < N:
        if proposal[i] < cumulative_weight[j]:
            index[i] = j
            i = i + 1
        else:
            j = j + 1

    return index
```

## Why Particle Filtering Can Estimate the Likelihood

The key problem is that the human's latent state $m_t$ is not observed. The
experimenter observes states, feedback, and actions, but not the internal
learning state that produced those actions. The exact likelihood must therefore
average over every plausible latent trajectory. For nonlinear and noisy learning
models, this average is not available in closed form.

Particle filtering solves this by representing uncertainty over $m_t$ with a
finite collection of simulated latent states. Each particle is one possible
history of the participant's latent state. At each trial, the model first
propagates every particle through the latent transition rule. This gives a Monte
Carlo approximation to the predictive distribution:

$$
p(m_t \mid a_{1:t-1}, s_{1:t}, \theta).
$$

Then the observed human action is used to score each particle. If a particle's
latent state makes the observed action likely, it receives a large weight. If it
makes the observed action unlikely, it receives a small weight. Averaging these
weights estimates the one-step predictive likelihood of the action:

$$
p(a_t \mid a_{1:t-1}, s_{1:t}, \theta).
$$

Because the full trajectory likelihood factorizes into a product of one-step
predictive likelihoods, summing the log of these estimated trial likelihoods
gives an estimate of the full trajectory log likelihood:

$$
\log p(a_{1:T} \mid s_{1:T}, \theta)
\approx
\sum_{t=1}^T
\log \left[
\frac{1}{N} \sum_{i=1}^N
p(a_t \mid m_t^{(i)}, s_t, a_{t-1}, \theta)
\right].
$$

Resampling is needed because, after several trials, most particles can receive
nearly zero weight. Without resampling, computation would be wasted on particles
that no longer represent plausible latent states for the observed behavior.
Systematic resampling keeps particles with high posterior support and removes
particles with low support, while still maintaining a population of $N$
particles. This allows the particle set to track the posterior distribution over
latent variables across the sequence.

In the current implementation, resampling happens after the likelihood
contribution for each trial is accumulated. This is important: the mean of the
pre-resampling weights estimates the marginal likelihood of the current action,
while the resampled particles provide an approximate filtering distribution for
the next trial.

The estimate becomes more accurate as the number of particles increases. With a
small number of particles, the likelihood estimate can be noisy because the
empirical particle distribution is a rough approximation. With more particles,
the empirical distribution better approximates the true filtering distribution,
and the estimated likelihood becomes more stable. The particle-filter estimate
of the likelihood is a Monte Carlo estimate; after taking the logarithm, the
estimated log likelihood is generally noisy and biased relative to the exact log
likelihood for finite $N$. This is why `estimate_nll_pf` uses `n_particles` as a
tuning parameter and why optimization with particle filters is often treated as
noisy likelihood optimization.

For fitting human behavior, the particle filter is useful because it handles
latent variables that are:

- stochastic, such as noisy Q-learning updates;
- sequentially dependent, because today's latent state depends on yesterday's;
- analytically intractable, because the integral over $m_{1:T}$ cannot be
  simplified into a closed-form expression;
- directly tied to action likelihood, because each latent particle can be scored
  by the behavioral policy.

Thus, the particle filter provides a practical bridge between a psychologically
meaningful latent learning model and the likelihood-based fitting objective
needed for parameter estimation.
