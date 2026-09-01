# Data Schema

The repository expects one timestamp-aligned CSV file containing the target wind-power signal and the meteorological/SCADA predictors used by the experiment.

## Required default columns

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | datetime | Observation timestamp |
| `wind_speed` | numeric | Wind-speed measurement |
| `wind_direction` | numeric | Wind-direction measurement |
| `temperature` | numeric | Ambient/related temperature |
| `humidity` | numeric | Relative humidity or corresponding humidity measurement |
| `power` | numeric | Wind-power output; default forecasting target |

The names can be changed in `config.yaml`.

## Input rules

1. Rows must represent chronological observations.
2. The timestamp column must be parseable by pandas.
3. Duplicate timestamps are removed by retaining the last occurrence after sorting.
4. Feature and target columns must be numeric or coercible to numeric.
5. Missing values are interpolated using time-aware interpolation when possible.
6. Interpolation is restricted by `interpolation_limit`.
7. Rows containing unresolved required-field missingness are removed when `drop_remaining_nan: true`.
8. Optional resampling can be enabled with a pandas-compatible rule such as `10min`, `30min`, or `1h`.
9. Resampling must only be configured if it matches the genuine experimental protocol.

## Leakage prevention

Normalization is fitted on each training fold only.

Validation and test values are transformed using training-fold statistics.

A forecast sample with origin time `t` uses historical observations from:

```text
[t-lookback+1, ..., t]
```

to predict:

```text
[t+1, ..., t+horizon]
```

No future target value is provided to the model through preprocessing.

## Dataset details that must be added to the manuscript

The current reproducibility package deliberately does not invent values that are absent from the manuscript.

Before final submission, document the real:

- dataset/source name;
- public URL or access procedure where applicable;
- SCADA system/farm provenance;
- observation start and end dates;
- sampling resolution;
- original sample count;
- retained sample count after cleaning;
- feature list;
- target definition and units;
- installed-capacity normalization, if used;
- missing-data percentage;
- interpolation policy;
- forecasting horizon;
- training/validation/test split boundaries.
