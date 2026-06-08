# Uber Fare Prediction using Regression Analysis

A machine learning project where I built a regression model to predict Uber ride fares — from raw data all the way to a working prediction system.


## What This Project Is About

The question I tried to answer here is simple: given a few basic details about an Uber ride — where it starts, where it ends, what time it is, and how many passengers — can a model accurately predict what the fare should be?

I went through the full process: understanding the business problem, exploring and cleaning the dataset, engineering new features, training four different models, comparing their performance, and picking the best one.

The short answer: yes, it works well. Random Forest and XGBoost both hit around R² ≈ 0.86–0.87 on unseen data. Trip distance is by far the strongest signal, followed by time of day. Passenger count barely matters — which actually makes sense, since Uber charges per trip, not per person.


## Project Structure


uber-fare-prediction/
│
├── uber_fare_prediction.ipynb   # Main notebook — full workflow start to finish
│
├── data/
│   └── uber.csv                 # Raw dataset (200k NYC Uber rides)
│
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md




## Dataset

The dataset contains ~200,000 historical Uber rides from New York City.

| Column | Description |
|---|---|
| `fare_amount` | Target — the actual fare paid |
| `pickup_datetime` | Date and time the ride started |
| `pickup_latitude` / `pickup_longitude` | GPS coordinates of pickup |
| `dropoff_latitude` / `dropoff_longitude` | GPS coordinates of drop-off |
| `passenger_count` | Number of passengers (1–6 after cleaning) |



## What I Did — Step by Step

### 1. Business Understanding
Mapped out why fare prediction actually matters for Uber — revenue protection, customer trust, driver positioning, and surge pricing decisions.

### 2. Exploratory Data Analysis (EDA)
- Checked for duplicates, missing values, invalid GPS coordinates
- Explored fare distribution — most rides are low-to-medium fare, with a few high-value outliers
- Found a clear time-of-day pattern — two spikes at morning and evening commute hours. This confirmed that hour is a real signal worth building a feature for.

### 3. Data Cleaning
- Removed fares ≤ 0 (clearly erroneous)
- Removed invalid passenger counts (0 or > 6)
- Dropped rows with missing values
- Removed duplicate rows
- Filtered out GPS coordinates outside valid Earth ranges
- Used IQR method to remove extreme fare outliers
- Dropped `Unnamed: 0` (index artifact) and `key` (unique ID with no predictive value)
- Removed trips where calculated distance = 0 km or > 100 km

### 4. Feature Engineering
This is where most of the model improvement came from.

| Feature | How It Was Created |
| `distance_km` | Haversine formula from pickup/dropoff GPS coordinates |
| `hour` | Extracted from `pickup_datetime` |
| `day` | Extracted from `pickup_datetime` |
| `month` | Extracted from `pickup_datetime` |
| `weekday` | Extracted from `pickup_datetime` |
| `is_peak_hour` | Binary flag: 1 if 7–10 AM or 4–8 PM, else 0 |

`distance_km` turned out to be the single most important feature by a significant margin.

### 5. Model Building
Tested four models:
- **Linear Regression** — simple baseline, couldn't handle non-linearity
- **Decision Tree** — better, but overfits
- **Random Forest** — strong, well-generalised
- **XGBoost** — highest accuracy overall

Used 80/20 train-test split and StandardScaler for feature normalisation.

### 6. Evaluation

| Model | MAE | RMSE | R² Score |

| Linear Regression | ~3.8 | ~5.4 | ~0.65 |
| Decision Tree | ~3.1 | ~4.7 | ~0.74 |
| Random Forest | ~2.2 | ~3.5 | **~0.86** |
| XGBoost | ~2.1 | ~3.4 | **~0.87** |

5-fold cross-validation confirmed results were stable across different splits — not just a lucky split.

Residual analysis showed errors centred around zero with no systematic bias.

### 7. Feature Importance

| Rank | Feature | Importance |

| 1 | `distance_km` | ~55–65% |
| 2 | `hour` | ~10–15% |
| 3 | `is_peak_hour` | ~8–12% |
| 4 | `weekday` / `day` | ~5–8% |
| 5 | `passenger_count` | ~1–3% |

---

## Sample Prediction

Ran the trained Random Forest model on a brand-new ride not in the training data:

- Pickup: Midtown Manhattan (40.7614°N, 73.9798°W)
- Drop-off: Near JFK Airport (40.6513°N, 73.8803°W)
- Passengers: 2
- Distance: 15.2 km
- Time: 6:00 PM Wednesday (peak hour)

The predicted fare came back in a range consistent with what Uber would actually charge for that route.


## How to Run This

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/uber-fare-prediction.git
cd uber-fare-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Open the notebook**
```bash
jupyter notebook uber_fare_prediction.ipynb
```

The dataset is already in the `data/` folder. The notebook will load it from `data/uber.csv` — no path changes needed.



## Dependencies


pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```


## Limitations

- No real-time traffic data — long detours due to congestion can push fares up in ways the model can't see
- No weather data — rain and storms spike demand and affect pricing
- No live surge multiplier — the model can't know what Uber's actual demand-supply ratio is at prediction time
- Haversine gives straight-line distance, not actual road distance


## Possible Improvements

- Integrate Google Maps API for actual driving distance and estimated time
- Pull weather data for pickup time and location
- Add real-time demand signals (driver availability, active ride requests)
- Set up monthly retraining pipeline so the model doesn't go stale


