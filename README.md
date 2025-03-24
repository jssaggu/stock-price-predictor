# Stock Price Predictor

An AI-powered stock price prediction tool that uses RSI (Relative Strength Index) and other technical indicators to analyze and predict stock prices.

## Features

- RSI-based technical analysis
- Multi-stock analysis support
- Configurable prediction parameters
- JSON output format
- Support for major tech stocks

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jssaggu/stock-price-predictor.git
cd stock-price-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

Run the analysis for one or more stocks:

```bash
python src/main.py --stocks AAPL NVDA AMD --output results.json
```

### Command Line Arguments

- `--stocks`: Space-separated list of stock symbols to analyze
- `--output`: Path to save the analysis results (default: results.json)

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Analysis thresholds
- Time horizons
- Confidence levels

## Project Structure

```
stock-price-predictor/
├── config/
│   └── config.yaml
├── src/
│   ├── agent/
│   │   └── stock_agent.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.9+
- yfinance
- pandas
- numpy
- openai
- pyyaml

## License

MIT License

## Author

Jasvinder Singh Saggu 