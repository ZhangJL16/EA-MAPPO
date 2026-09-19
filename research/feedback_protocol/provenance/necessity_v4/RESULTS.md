# Final pre-B2 necessity study

Decision under frozen operational rule: **GO**.

Quality condition: True; compute-growth condition: True.
Exact-reference statuses: `{'exact': 10, 'unresolved': 2}`.
Fixed-policy evaluations: `{'exact_conditional_policy': 371}` (fresh subset `{'exact_conditional_policy': 240}`).

## Fresh DEV: large-tier episode-budget exact risk

All numbers integrate feedback exactly conditional on policy seed. U = unresolved (not zero).

| Root | Condition | Bayes optimum | Cover | PS | Beam | VOI | MCTS |
|---|---|---:|---:|---:|---:|---:|---:|
| hierarchy-2201 | weak_information | 8.062500 | 10.022500 | 8.750000 | 8.062500 | 8.062500 | 9.750000 |
| hierarchy-2202 | redundant_channels | 6.641538 | 10.380000 | 8.538462 | 6.641538 | 6.960000 | 8.723077 |
| hierarchy-2201 | long_horizon | 8.280000 | 9.900000 | 12.937500 | 12.153750 | 9.191250 | 16.000000 |
| hierarchy-2201 | deeper_hierarchy | U | 16.675071 | 15.464286 | 15.785714 | 14.785714 | 16.677679 |
| hierarchy-2201 | base | 6.937500 | 8.760000 | 8.750000 | 6.937500 | 6.937500 | 9.750000 |
| hierarchy-2202 | long_horizon | 8.280000 | 10.710000 | 13.312500 | 8.280000 | 8.280000 | 14.531250 |
| hierarchy-2202 | base | 7.140000 | 9.570000 | 8.687500 | 7.140000 | 7.140000 | 9.606250 |
| hierarchy-2201 | redundant_channels | 7.140000 | 10.371474 | 8.947368 | 7.140000 | 7.357895 | 10.032105 |
| hierarchy-2201 | costly_acquisition | 8.062500 | 12.000000 | 8.750000 | 8.062500 | 8.062500 | 9.812500 |
| hierarchy-2202 | deeper_hierarchy | U | 16.612875 | 15.718750 | 15.187500 | 15.187500 | 15.864062 |
| hierarchy-2202 | costly_acquisition | 8.250000 | 10.728750 | 8.687500 | 8.250000 | 8.250000 | 8.937500 |
| hierarchy-2202 | weak_information | 8.250000 | 10.530000 | 8.687500 | 8.250000 | 8.250000 | 9.587500 |

## Legacy DEV deterministic-policy re-evaluation

This is the new deterministic-work contract, not a claim to reproduce every clock-dependent old action.

| Root | B | H | Method | Tier | Seed | Exact risk | Status |
|---|---:|---:|---|---|---:|---:|---|
| iid-dev-1 | 5 | 4 | channel_cover | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | channel_cover | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | posterior_sampling | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | posterior_sampling | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | posterior_sampling | large | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | posterior_sampling | small | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | beam_bayes | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | beam_bayes | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | one_step_voi | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | one_step_voi | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 4 | cached_bayes | reference | 0 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | channel_cover | large | 11 | 4.800000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | channel_cover | small | 11 | 4.800000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | posterior_sampling | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | posterior_sampling | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | posterior_sampling | large | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | posterior_sampling | small | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | beam_bayes | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | beam_bayes | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | one_step_voi | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | one_step_voi | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 5 | 8 | cached_bayes | reference | 0 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | channel_cover | large | 11 | 6.200000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | channel_cover | small | 11 | 6.200000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | posterior_sampling | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | posterior_sampling | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | posterior_sampling | large | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | posterior_sampling | small | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | beam_bayes | large | 11 | 3.500000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | beam_bayes | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | one_step_voi | large | 11 | 3.500000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | one_step_voi | small | 11 | 3.500000 | exact_conditional_policy |
| iid-dev-0 | 3 | 8 | cached_bayes | reference | 0 | 3.500000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | channel_cover | large | 11 | 10.370400 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | channel_cover | small | 11 | 10.370400 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | posterior_sampling | large | 11 | 6.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | posterior_sampling | small | 11 | 6.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | posterior_sampling | large | 12 | 6.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | posterior_sampling | small | 12 | 6.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | beam_bayes | large | 11 | 5.250000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | beam_bayes | small | 11 | 6.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | one_step_voi | large | 11 | 5.250000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | one_step_voi | small | 11 | 6.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 12 | cached_bayes | reference | 0 | 5.250000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | channel_cover | large | 11 | 5.300000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | channel_cover | small | 11 | 5.300000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | posterior_sampling | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | posterior_sampling | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | posterior_sampling | large | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | posterior_sampling | small | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | beam_bayes | large | 11 | 3.500000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | beam_bayes | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | one_step_voi | large | 11 | 3.500000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | one_step_voi | small | 11 | 3.500000 | exact_conditional_policy |
| iid-dev-0 | 5 | 8 | cached_bayes | reference | 0 | 3.500000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | channel_cover | large | 11 | 6.400000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | channel_cover | small | 11 | 6.400000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | posterior_sampling | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | posterior_sampling | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | posterior_sampling | large | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | posterior_sampling | small | 12 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | beam_bayes | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | beam_bayes | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | one_step_voi | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | one_step_voi | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 8 | cached_bayes | reference | 0 | 4.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | channel_cover | large | 11 | 3.400000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | channel_cover | small | 11 | 3.400000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | posterior_sampling | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | posterior_sampling | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | posterior_sampling | large | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | posterior_sampling | small | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | beam_bayes | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | beam_bayes | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | one_step_voi | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | one_step_voi | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-1 | 3 | 4 | cached_bayes | reference | 0 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | channel_cover | large | 11 | 3.450000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | channel_cover | small | 11 | 3.450000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | posterior_sampling | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | posterior_sampling | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | posterior_sampling | large | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | posterior_sampling | small | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | beam_bayes | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | beam_bayes | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | one_step_voi | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | one_step_voi | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 3 | 4 | cached_bayes | reference | 0 | 2.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | channel_cover | large | 11 | 7.166760 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | channel_cover | small | 11 | 6.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | posterior_sampling | large | 11 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | posterior_sampling | small | 11 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | posterior_sampling | large | 12 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | posterior_sampling | small | 12 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | beam_bayes | large | 11 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | beam_bayes | small | 11 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | one_step_voi | large | 11 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | one_step_voi | small | 11 | 4.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 8 | cached_bayes | reference | 0 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | channel_cover | large | 11 | 7.187000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | channel_cover | small | 11 | 7.187000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | posterior_sampling | large | 11 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | posterior_sampling | small | 11 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | posterior_sampling | large | 12 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | posterior_sampling | small | 12 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | beam_bayes | large | 11 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | beam_bayes | small | 11 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | one_step_voi | large | 11 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | one_step_voi | small | 11 | 4.000000 | exact_conditional_policy |
| scale-sparse-dev-0 | 7 | 8 | cached_bayes | reference | 0 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | channel_cover | large | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | channel_cover | small | 11 | 4.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | posterior_sampling | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | posterior_sampling | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | posterior_sampling | large | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | posterior_sampling | small | 12 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | beam_bayes | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | beam_bayes | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | one_step_voi | large | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | one_step_voi | small | 11 | 2.000000 | exact_conditional_policy |
| iid-dev-0 | 5 | 4 | cached_bayes | reference | 0 | 2.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | channel_cover | large | 11 | 7.833800 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | channel_cover | small | 11 | 8.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | posterior_sampling | large | 11 | 6.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | posterior_sampling | small | 11 | 6.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | posterior_sampling | large | 12 | 6.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | posterior_sampling | small | 12 | 6.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | beam_bayes | large | 11 | 6.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | beam_bayes | small | 11 | 6.000000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | one_step_voi | large | 11 | 5.600000 | exact_conditional_policy |
| scale-dense-dev-0 | 7 | 12 | one_step_voi | small | 11 | 6.000000 | exact_conditional_policy |

## Boundaries

- Two fresh root groups; axis variants are repeated conditions, not independent roots.
- Exact expected values integrate environment feedback only, conditional on each frozen algorithm seed.
- MCTS/PS here do not integrate the distribution over all internal random seeds.
- Wall watchdog invalidates a result; it never selects an action.
- MC fallbacks remain MC and cannot satisfy the exact-gap GO criterion.
- No new B2 gate, neural training, final test or external task is executed.
