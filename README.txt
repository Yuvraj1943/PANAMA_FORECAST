Put these four files in this folder before re-training or running the tests:
  continuous_dataset.csv            (hourly Panama demand, weather, calendar; 2015-2020)
  weekly_pre-dispatch_forecast.csv  (CND's official weekly forecast - the benchmark)
  train_dataframes.xlsx             (14 walk-forward training sets, one sheet per week)
  test_dataframes.xlsx              (14 matching 168-hour test weeks)
They were not included in the zip because of their size (about 55 MB). The dashboard (app.py) does NOT need them.
