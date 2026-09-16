#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define HOCBF_EXPORT __declspec(dllexport)
#else
#define HOCBF_EXPORT __attribute__((visibility("default")))
#endif

/*
 * Native implementation of the three-variable Hildreth/Dykstra loop used by
 * project_polyhedral_qp. Validation, matrix inversion, degenerate-row checks,
 * and result construction remain in Python. Keeping this loop mechanically
 * aligned with the reference implementation preserves solver semantics while
 * removing millions of Python scalar operations.
 */
HOCBF_EXPORT int hocbf_project_qp3(
    const double *center,
    const double *inverse,
    const double *matrix,
    const double *bounds,
    const double *denominators,
    int64_t constraint_count,
    int max_iterations,
    double tolerance,
    int stagnation_sweeps,
    double *acceleration,
    double *multipliers,
    int *iterations,
    int *converged,
    int *stagnated,
    double *maximum_violation,
    int *active_constraints) {
  double *violation_history;
  int sweep;

  if (constraint_count < 0 || max_iterations <= 0 || stagnation_sweeps <= 0 ||
      tolerance <= 0.0) {
    return 1;
  }
  violation_history = (double *)calloc((size_t)max_iterations, sizeof(double));
  if (violation_history == NULL) {
    return 2;
  }
  memcpy(acceleration, center, 3 * sizeof(double));
  memset(multipliers, 0, (size_t)constraint_count * sizeof(double));
  *iterations = 0;
  *converged = 0;
  *stagnated = 0;

  for (sweep = 1; sweep <= max_iterations; ++sweep) {
    double largest_delta = 0.0;
    double max_violation = 0.0;
    int64_t index;

    for (index = 0; index < constraint_count; ++index) {
      const double denominator = denominators[index];
      const double *row;
      double violation;
      double candidate;
      double delta;
      double transformed0;
      double transformed1;
      double transformed2;
      if (denominator <= 1e-15) {
        continue;
      }
      row = matrix + 3 * index;
      violation = bounds[index] -
                  (row[0] * acceleration[0] + row[1] * acceleration[1] +
                   row[2] * acceleration[2]);
      candidate = multipliers[index] + violation / denominator;
      if (candidate < 0.0) {
        candidate = 0.0;
      }
      delta = candidate - multipliers[index];
      if (delta != 0.0) {
        transformed0 = inverse[0] * row[0] + inverse[1] * row[1] +
                       inverse[2] * row[2];
        transformed1 = inverse[3] * row[0] + inverse[4] * row[1] +
                       inverse[5] * row[2];
        transformed2 = inverse[6] * row[0] + inverse[7] * row[1] +
                       inverse[8] * row[2];
        acceleration[0] += delta * transformed0;
        acceleration[1] += delta * transformed1;
        acceleration[2] += delta * transformed2;
        multipliers[index] = candidate;
        if (fabs(delta) > largest_delta) {
          largest_delta = fabs(delta);
        }
      }
    }

    for (index = 0; index < constraint_count; ++index) {
      const double *row = matrix + 3 * index;
      const double violation =
          bounds[index] -
          (row[0] * acceleration[0] + row[1] * acceleration[1] +
           row[2] * acceleration[2]);
      if (violation > max_violation) {
        max_violation = violation;
      }
    }
    violation_history[sweep - 1] = max_violation;
    *iterations = sweep;
    if (max_violation <= tolerance && largest_delta <= tolerance) {
      *converged = 1;
      break;
    }
    if (sweep > stagnation_sweeps &&
        max_violation > fmax(tolerance * 10.0, 1e-3) &&
        max_violation >=
            violation_history[sweep - stagnation_sweeps - 1] * (1.0 - 1e-6) -
                tolerance) {
      *stagnated = 1;
      break;
    }
  }

  *maximum_violation = 0.0;
  *active_constraints = 0;
  for (int64_t index = 0; index < constraint_count; ++index) {
    const double *row = matrix + 3 * index;
    const double violation =
        bounds[index] -
        (row[0] * acceleration[0] + row[1] * acceleration[1] +
         row[2] * acceleration[2]);
    if (violation > *maximum_violation) {
      *maximum_violation = violation;
    }
    if (multipliers[index] > tolerance) {
      *active_constraints += 1;
    }
  }
  free(violation_history);
  return 0;
}
