# riskEntryManagement
ML Dynamic Risk &amp; Entry Management
This project implements an **end-to-end machine learning pipeline** for trading signal evaluation using event-based labeling.  
It works with two datasets:

- **Bars Data (`bars_df`)** → OHLCV (Open, High, Low, Close, Volume) market data.  
- **Signals Data (`signals_df`)** → Trading signals aligned with timestamps.  

The pipeline labels signals, extracts features, and trains a classification model to predict trade outcomes.

---

## 📂 Project Structure

---

## ⚙️ How It Works

1. **Load Data**  
   - `load_bars()` → Generates sample OHLCV data.  
   - `load_signals()` → Generates random trading signals aligned with bars.  

2. **Feature Engineering**  
   - Computes rolling statistics, moving averages, returns, etc.  

3. **Label Generation**  
   - Labels each signal as:
     - `1` → Price went up in lookahead window  
     - `-1` → Price went down  
     - `0` → Neutral  

4. **Model Training**  
   - Uses scikit-learn classifiers (Logistic Regression, Random Forest, etc.).  
   - Splits data into train/test sets.  
   - Prints evaluation metrics.  

---

## 🚀 Usage

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
python main.py

Bars sample:
             timestamp        open        high         low       close  volume
0 2024-01-01 00:00:00  100.21      101.45      98.92      100.87     562
1 2024-01-01 01:00:00  101.01      102.15      99.98      101.34     487
...

Signals sample:
             timestamp  signal
0 2024-01-01 00:00:00       0
1 2024-01-01 01:00:00       1
...

Classification Report:
              precision    recall  f1-score   support
         -1       0.61      0.58      0.59        50
          0       0.62      0.65      0.63        47
          1       0.64      0.66      0.65        53


