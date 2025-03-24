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
    def __init__(self):
        self.config = self._load_config()
        self._setup_openai()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_openai(self):
        """Set up OpenAI API."""
        load_dotenv()
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
    
    def _calculate_rsi(self, data, periods=14):
        """Calculate RSI for the given data."""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_stock_data(self, symbol):
        """Get historical stock data and current price."""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            current_price = stock.info.get('regularMarketPrice', 0)
            return hist, current_price
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")
    
    def _prepare_context(self, symbol, hist_data, current_price):
        """Prepare context for the prediction model."""
        # Calculate RSI
        rsi = self._calculate_rsi(hist_data).iloc[-1]
        
        # Get recent price changes
        recent_prices = hist_data['Close'].tail(5)
        price_changes = recent_prices.pct_change()
        
        # Prepare market data
        market_data = {
            'current_price': current_price,
            'rsi': rsi,
            'price_changes': price_changes.tolist(),
            'volume': hist_data['Volume'].tail(5).mean(),
            'volatility': hist_data['Close'].pct_change().std()
        }
        
        return market_data
    
    def predict(self, symbol):
        """Generate a prediction for a given stock symbol."""
        try:
            # Get stock data
            hist_data, current_price = self._get_stock_data(symbol)
            
            # Prepare context
            market_data = self._prepare_context(symbol, hist_data, current_price)
            
            # Prepare prompt for the model
            prompt = (
                f"Based on the following market data for {symbol}, provide a price prediction:\n"
                f"Current Price: ${market_data['current_price']:.2f}\n"
                f"RSI: {market_data['rsi']:.2f}\n"
                "Recent Price Changes: " + str([f"{x:.2%}" for x in market_data['price_changes'] if pd.notnull(x)]) + "\n"
                f"Average Volume: {market_data['volume']:.0f}\n"
                f"Volatility: {market_data['volatility']:.2%}\n\n"
                "Please provide a prediction in EXACTLY this JSON format:\n"
                "{\n"
                '    "price": 000.00,\n'
                '    "confidence": 0.00,\n'
                '    "time_horizon": "X months",\n'
                '    "reasoning": "Your detailed analysis here"\n'
                "}"
            )
            
            # Get prediction from OpenAI
            response = openai.chat.completions.create(
                model=self.config['openai']['model'],
                messages=[
                    {"role": "system", "content": "You are a stock market analyst. Always respond with valid JSON containing exactly these keys: price (float), confidence (float 0-1), time_horizon (string), reasoning (string)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse the response
            prediction_text = response.choices[0].message.content
            try:
                # Try to parse as JSON first
                prediction = json.loads(prediction_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract the values using string manipulation
                import re
                
                # More robust regex patterns
                price_pattern = r'["\']?price["\']?\s*:\s*(\d+\.?\d*)'
                confidence_pattern = r'["\']?confidence["\']?\s*:\s*(\d*\.?\d*)'
                time_horizon_pattern = r'["\']?time_horizon["\']?\s*:\s*["\']([^"\']+)["\']'
                reasoning_pattern = r'["\']?reasoning["\']?\s*:\s*["\']([^"\']+)["\']'
                
                price_match = re.search(price_pattern, prediction_text, re.IGNORECASE)
                confidence_match = re.search(confidence_pattern, prediction_text, re.IGNORECASE)
                time_horizon_match = re.search(time_horizon_pattern, prediction_text, re.IGNORECASE)
                reasoning_match = re.search(reasoning_pattern, prediction_text, re.IGNORECASE)
                
                prediction = {
                    'price': float(price_match.group(1)) if price_match else current_price * 1.05,  # 5% increase as fallback
                    'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
                    'time_horizon': time_horizon_match.group(1) if time_horizon_match else "2 months",
                    'reasoning': reasoning_match.group(1) if reasoning_match else prediction_text
                }
            
            # Add RSI to the prediction
            prediction['rsi'] = market_data['rsi']
            
            return prediction
            
        except Exception as e:
            raise Exception(f"Error generating prediction for {symbol}: {str(e)}")
    
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

    def analyze_stocks(self, symbols: List[str]) -> Dict[str, Dict]:
        """Analyze multiple stocks and return predictions."""
        predictions = {}
        for symbol in symbols:
            prediction = self._generate_prediction(symbol)
            if prediction:
                predictions[symbol] = prediction
        return predictions
    
    def _generate_prediction(self, symbol: str) -> Optional[Dict]:
        """Generate prediction using OpenAI API."""
        hist, current_price = self._get_stock_data(symbol)
        if hist is None or current_price is None:
            return None
            
        rsi = self._calculate_rsi(hist)
        context = self._prepare_context(symbol, hist, current_price)
        
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