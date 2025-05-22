# <center>ParallelTime</center>

### Welcome to the official repository of: ParallelTime: Dynamically Weighting the Balance of Short- and Long-Term Temporal Dependencies. 
## Usage

1. Obtain the fundamental long-term forecasting datasets, including Weather, Illness, Traffic, Electricity, and ETT (comprising 4 datasets)
You can do it by downloading it from this [link](https://drive.google.com/drive/folders/13WaWueE6w7f5_4_rlo-5ABZ_g9NDWM9E?usp=sharing), and move them into dataset/ directory 
```
├── dataset
│   ├── electricity
│   │   └── electricity.csv
│   ├── ETT-small
│   │   ├── ETTh1.csv
│   │   ├── ETTh2.csv
│   │   ├── ETTm1.csv
│   │   └── ETTm2.csv
│   ├── illness
│   │   └── national_illness.csv
│   ├── traffic
│   │   └── traffic.csv
│   └── weather
│       └── weather.csv 
```


2. Install requirements. ```pip install -r requirements.txt``` with python version `3.11.9`.

3. Look through our scripts located at ```./scripts``` which run 4 runs for diffrent prediction length depend on the dataset. You'll find the core of ParallelTime in ```models/ParallelTime.py```. For example, to get the multivariate forecasting results for ETTh1 dataset, just run the following command `sh ./scripts/etth1.sh`

For the other datasets run on a Linux machine:
```
sh ./scripts/electricity.sh
sh ./scripts/etth2.sh
sh ./scripts/ettm2.sh
sh ./scripts/traffic.sh
sh ./scripts/etth1.sh
sh ./scripts/ettm1.sh
sh ./scripts/illness.sh
sh ./scripts/weather.sh
```
After the run ends for each prediction length, you can open ```./result_long_term_forecast_ETTh1.txt``` to view the results once the model run is complete. Additionally, logs and configurations are saved for each run in ```./multirun/current_date/ETTh1/run_time```, which can be used to review more detailed results.

<img src="figures/model-architucture.drawio.png" alt="ParallelTime Architucture" width="75%">

## Acknowledgement

We are deeply grateful for the valuable code and efforts contributed by the following GitHub repositories.
- iTransformer (https://github.com/thuml/iTransformer)
- PatchTST (https://github.com/yuqinie98/PatchTST)