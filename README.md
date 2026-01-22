# PairTrading_RL: Reinforcement Learning for Pair Trading

A reinforcement learning-based trading system that uses Actor-Critic algorithms (A2C) to train agents for algorithmic pair trading strategies on cryptocurrency markets.

## Project Overview

This project implements a pair trading strategy using deep reinforcement learning (DRL). The system:

- Trains RL agents to trade correlated cryptocurrency pairs (BTC/EUR, BTC/GBP)
- Uses the A2C algorithm from Stable Baselines3
- Provides custom Gymnasium environment for trading simulation
- Includes data preprocessing, model training, and backtesting capabilities

## Project Structure

```
PairTrading_RL/
├── RL/                           # Main RL training and testing
│   ├── trainer.py               # Script to train A2C models
│   ├── tester.py                # Backtesting and evaluation of trained models
│   ├── environment_A.py          # Custom Gymnasium trading environment
│   ├── data_preprocessor.py      # Data preprocessing utilities
│   ├── models/                   # Trained model checkpoints (A2C-v1 to v16+)
│   └── data/                     # Training/testing datasets
├── data/                         # Raw and processed cryptocurrency data
│   ├── btceur_1min.csv          # BTC/EUR 1-minute OHLCV data
│   ├── btcgbp_1min.csv          # BTC/GBP 1-minute OHLCV data
│   └── ...
├── data_miner/                   # Data collection scripts
│   ├── binance_data.py
│   ├── yfinance_data.py
│   └── tiingo_data.py
├── pair_tester.py               # Pair correlation and cointegration analysis
└── README.md
```

## Requirements

- Python 3.8+
- gymnasium
- stable-baselines3
- pandas
- numpy
- matplotlib
- statsmodels

## Installation

```bash
# Install dependencies
pip install gymnasium stable-baselines3 pandas numpy matplotlib statsmodels scikit-learn

# Optional: For additional data collection features
pip install yfinance tiingo binance-connector
```

## Training a New Model

### Quick Start

To continue training or start a new model:

```bash
cd RL
python trainer.py
```

### Training Configuration

Edit `trainer.py` to customize training:

```python
# Model hyperparameters (around line 45-58)
model = A2C(
    'MultiInputPolicy',
    env=env,
    learning_rate=1e-4,        # Learning rate (try 1e-4 to 5e-4)
    n_steps=128,               # Steps per update
    gamma=0.99,                # Discount factor
    gae_lambda=0.95,           # Advantage estimation parameter
    ent_coef=0.01,             # Entropy bonus for exploration
    vf_coef=0.5,               # Value function weight
    max_grad_norm=0.5,         # Gradient clipping
    seed=42
)

# Training parameters
model.learn(total_timesteps=100000, reset_num_timesteps=False)

# Model checkpoint naming
model_name = 'A2C_rw100NEW-v1'  # Change prefix for different experiments
```

### Key Training Options

- **Data Selection**: Line 33

  ```python
  df = pd.read_csv('data/final_normal_rw100.csv')[:-30000]
  ```

  Choose dataset and train/test split

- **Reward Structure**: Different rolling window datasets
  - `final_normal_rw100.csv` - 100-step rolling window rewards
  - `final_normal_rw900.csv` - 900-step rolling window rewards

- **Model Prefix**: Line 31
  ```python
  model_name = 'A2C_rw100NEW-v1'  # Customize for different experiments
  ```

### Resuming Training

The trainer automatically:

- Detects existing models and loads them
- Continues training from the last checkpoint
- Creates versioned backups (v1, v2, v3, etc.)

## Testing & Backtesting

### Run Backtest on Trained Model

```bash
cd RL
python tester.py
```

### Customize Testing

In `tester.py`, specify:

- Model to test
- Test dataset
- Stop-loss/take-profit strategies

### Backtest Metrics

The tester calculates:

- **Total Trades**: Number of closed positions
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Mean profit/loss per trade
- **Profit Factor**: Ratio of gross profit to gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Total Return %**: Percentage return on initial capital
- **Trade Log**: Detailed per-trade metrics

## Custom Trading Environment

The `PairTradingEnv` (in `environment_A.py`) implements a Gymnasium environment with:

### State Space (Observations)

- **Position**: Current trading position (-1 short, 0 neutral, 1 long)
- **Z-Score**: Statistical measure of pair divergence
- **Current Returns**: Ongoing P&L metrics

### Action Space

- **0**: Short (sell)
- **1**: Hold (maintain position)
- **2**: Long (buy)

### Custom Initialization

```python
from environment_A import PairTradingEnv
import gymnasium as gym
from gymnasium import register

env_id = "PairTradingEnv-v0"
register(env_id, entry_point=PairTradingEnv)

# Use custom data
env = gym.make(env_id, df=custom_dataframe)
```

## Data Preprocessing

### Clean and Prepare Data

```bash
cd data
jupyter notebook cleaner.ipynb
```

The preprocessing pipeline:

1. Handles missing values (imputation)
2. Removes duplicates
3. Calculates zscore and rolling metrics
4. Generates training/testing splits

## Available Models

Pre-trained models in `RL/models/`:

- `A2C_rw100NEW-v1` to `v16`: Models trained on 100-step rolling window rewards
- `A2C_rw900NEW-v1` to `v7`: Models trained on 900-step rolling window rewards

Load and use:

```python
from stable_baselines3 import A2C

model = A2C.load('RL/models/A2C_rw100NEW-v16', env=env)
predictions, _ = model.predict(observations)
```

## Common Workflows

### Workflow 1: Train New Experiment

```bash
# 1. Prepare data
cd data
jupyter notebook cleaner.ipynb

# 2. Update trainer.py with new parameters
# 3. Start training
cd ../RL
python trainer.py
```

### Workflow 2: Evaluate Model Performance

```bash
cd RL
python tester.py
# Review metrics and trade log output
```

### Workflow 3: Switch Datasets

```python
# In trainer.py, line 33:
df = pd.read_csv('data/final_normal_rw900.csv')[:-30000]  # Use different dataset
# Update model_name accordingly
```

## Troubleshooting

### IndexError During Training

Occurs when reaching end of dataset. This is normal - the training exits gracefully and saves the model.

### CUDA/GPU Issues

- Training works on CPU by default
- For GPU support, ensure stable-baselines3 is configured for your CUDA version

### Data Loading Issues

- Verify CSV files exist in `RL/data/` directory
- Check file paths are relative to the RL directory
- Ensure data format matches expected columns

## Citation & References

- **Gymnasium**: https://gymnasium.farama.org/
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Pair Trading**: Statistical arbitrage strategy trading correlated assets

## License

[Add your license here]

## Contact

[Add contact information if needed]
