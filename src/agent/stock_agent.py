from typing import List, Dict, Any, Optional, Tuple
import yfinance as yf
from newsapi import NewsApiClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import os
from dotenv import load_dotenv
import openai
import json

class StockAgent:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the StockAgent with configuration."""
        self.config = self._load_config(config_path)
        self._setup_openai()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_openai(self):
        """Setup OpenAI API key."""
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
    
    def _calculate_rsi(self, data: pd.DataFrame, periods: int = 14) -> float:
        """Calculate RSI for the given data."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _get_stock_data(self, symbol: str) -> Tuple[pd.DataFrame, float]:
        """Get historical stock data and current price."""
        try:
            stock = yf.Ticker(symbol)
            # Get 1 year of daily data for RSI calculation
            hist = stock.history(period="1y")
            current_price = hist['Close'].iloc[-1]
            return hist, current_price
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None, None
    
    def _prepare_context(self, symbol: str, current_price: float, rsi: float) -> str:
        """Prepare context for the model."""
        context = f"""Analyze the stock {symbol} with the following information:
Current Price: ${current_price:.2f}
RSI: {rsi:.2f}

Please provide a prediction in the following format:
Price: [predicted price]
Confidence: [confidence level between 0 and 1]
Time Horizon: [predicted time to reach target in days/weeks/months]
Reasoning: [detailed explanation including RSI analysis]

Example:
Price: 150.00
Confidence: 0.75
Time Horizon: 3 months
Reasoning: The stock shows [RSI analysis] with current RSI of {rsi:.2f}. [Other factors]...

Please ensure your prediction is realistic and based on both technical indicators and market sentiment."""
        return context
    
    def _generate_prediction(self, symbol: str) -> Optional[Dict]:
        """Generate prediction using OpenAI API."""
        hist, current_price = self._get_stock_data(symbol)
        if hist is None or current_price is None:
            return None
            
        rsi = self._calculate_rsi(hist)
        context = self._prepare_context(symbol, current_price, rsi)
        
        client = openai.OpenAI()
        for model in self.config['openai']['models']:
            try:
                print(f"Trying model: {model}")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a stock market analyst providing price predictions with confidence levels and time horizons."},
                        {"role": "user", "content": context}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                print(f"Model response:\n{response.choices[0].message.content}")
                return self._parse_prediction(response.choices[0].message.content, rsi)
            except Exception as e:
                print(f"Error with model {model}: {str(e)}")
                continue
        return None
    
    def _parse_prediction(self, response: str, rsi: float) -> Dict:
        """Parse the model's response into structured data."""
        try:
            lines = response.strip().split('\n')
            prediction = {}
            
            # Initialize default values
            prediction['price'] = None
            prediction['confidence'] = 0.0
            prediction['time_horizon'] = "Unknown"
            prediction['reasoning'] = ""
            prediction['rsi'] = rsi
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'price':
                        try:
                            prediction['price'] = float(value.replace('$', '').replace(',', ''))
                        except:
                            continue
                    elif key == 'confidence':
                        try:
                            prediction['confidence'] = float(value)
                        except:
                            continue
                    elif key == 'time horizon':
                        prediction['time_horizon'] = value
                    elif key == 'reasoning':
                        prediction['reasoning'] = value
            
            # If we couldn't parse a price or confidence, return None
            if prediction['price'] is None or prediction['confidence'] == 0.0:
                return None
                
            return prediction
        except Exception as e:
            print(f"Error parsing prediction: {str(e)}")
            return None
    
    def analyze_stocks(self, symbols: List[str]) -> Dict[str, Dict]:
        """Analyze multiple stocks and return predictions."""
        predictions = {}
        for symbol in symbols:
            prediction = self._generate_prediction(symbol)
            if prediction:
                predictions[symbol] = prediction
        return predictions
    
    def _get_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch relevant news articles using News API
        """
        news = self.news_client.get_everything(
            q=symbol,
            language=self.config['news_api']['language'],
            sort_by=self.config['news_api']['sort_by'],
            page_size=self.config['news_api']['page_size']
        )
        return news['articles']
    
    def _generate_recommendation(self, prediction: Dict[str, Any]) -> str:
        """
        Generate buy/sell/hold recommendation based on prediction
        """
        if prediction['confidence'] < self.config['stock_analysis']['confidence_threshold']:
            return 'hold'
        
        if prediction['price'] is None:
            return 'hold'
        
        # This is a simplified recommendation logic - you might want to make it more sophisticated
        if prediction['confidence'] >= self.config['recommendations']['buy_threshold']:
            return 'buy'
        elif prediction['confidence'] <= self.config['recommendations']['sell_threshold']:
            return 'sell'
        else:
            return 'hold' 