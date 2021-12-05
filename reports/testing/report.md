## Coverage

Commands: 
```shell
coverage run -m pytest tests
```

```shell
coverage report -m
```

Coverage:
```shell
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
definitions.py                          21      0   100%
src/data/dataframes.py                  12      0   100%
src/data/download.py                    18      0   100%
src/features/feature_extraction.py      87      0   100%
src/models/parameter_selection.py       28     12    57%   12-43
src/models/threshold_tuning.py          55     25    55%   41-80
tests/test_data.py                      25      0   100%
tests/test_features.py                  49      0   100%
tests/test_threshold_tuning.py          14      0   100%
tests/test_validate_logreg.py           16      0   100%
------------------------------------------------------------------
TOTAL                                  325     37    89%
```
