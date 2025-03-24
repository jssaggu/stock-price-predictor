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
import re

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
    
    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI for a series of prices."""
        try:
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            
            # Calculate RS and RSI
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1]
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return 50  # Return neutral RSI on error
    
    def _get_stock_data(self, symbol):
        """Get historical stock data and current price."""
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            hist_data = stock.history(period="1y")
            
            if hist_data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Calculate current price
            current_price = hist_data['Close'].iloc[-1]
            
            # Calculate price changes
            price_changes = hist_data['Close'].pct_change().dropna().tolist()
            
            # Calculate average volume
            avg_volume = hist_data['Volume'].mean()
            
            # Calculate volatility
            volatility = hist_data['Close'].pct_change().std()
            
            return {
                'Close': hist_data['Close'],
                'current_price': current_price,
                'price_changes': price_changes,
                'volume': avg_volume,
                'volatility': volatility
            }
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _prepare_context(self, symbol, stock_data, rsi):
        """Prepare context for the model."""
        return (
            f"Based on the following market data for {symbol}, provide a price prediction:\n"
            f"Current Price: ${stock_data['current_price']:.2f}\n"
            f"RSI: {rsi:.2f}\n"
            "Recent Price Changes: " + str([f"{x:.2%}" for x in stock_data['price_changes'][-5:] if pd.notnull(x)]) + "\n"
            f"Average Volume: {stock_data['volume']:.0f}\n"
            f"Volatility: {stock_data['volatility']:.2%}\n\n"
            "Please provide a prediction in EXACTLY this JSON format:\n"
            "{\n"
            '    "price": 000.00,\n'
            '    "confidence": 0.00,\n'
            '    "time_horizon": "X months",\n'
            '    "reasoning": "Your detailed analysis here"\n'
            "}"
        )
    
    def predict(self, stock_symbol):
        """Generate a prediction for a given stock."""
        try:
            # Get stock data
            stock_data = self._get_stock_data(stock_symbol)
            if stock_data is None:
                raise ValueError(f"Could not fetch data for {stock_symbol}")

            # Calculate RSI
            rsi = self._calculate_rsi(stock_data['Close'])
            
            # Prepare context for the model
            context = self._prepare_context(stock_symbol, stock_data, rsi)
            
            # Try different models in order of preference
            models = self.config['openai']['models']
            last_error = None
            
            for model in models:
                try:
                    # Create OpenAI client
                    client = openai.OpenAI()
                    
                    # Generate prediction
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a stock market analyst. Provide price predictions in a strict JSON format with keys: price, confidence, time_horizon, reasoning"},
                            {"role": "user", "content": context}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    # Parse the response
                    try:
                        prediction = json.loads(response.choices[0].message.content)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try to extract values using regex
                        content = response.choices[0].message.content
                        price_match = re.search(r'\$(\d+\.?\d*)', content)
                        confidence_match = re.search(r'(\d+\.?\d*)%', content)
                        time_horizon_match = re.search(r'time horizon:?\s*([^.,]+)', content, re.IGNORECASE)
                        reasoning_match = re.search(r'reasoning:?\s*([^.,]+)', content, re.IGNORECASE)
                        
                        prediction = {
                            'price': float(price_match.group(1)) if price_match else stock_data['current_price'],
                            'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
                            'time_horizon': time_horizon_match.group(1).strip() if time_horizon_match else '1 month',
                            'reasoning': reasoning_match.group(1).strip() if reasoning_match else 'No reasoning provided'
                        }
                    
                    # Add RSI to the prediction
                    prediction['rsi'] = rsi
                    
                    # Add recommendation based on confidence
                    confidence = prediction['confidence']
                    if confidence >= self.config['recommendations']['buy_threshold']:
                        prediction['recommendation'] = 'BUY'
                    elif confidence <= self.config['recommendations']['sell_threshold']:
                        prediction['recommendation'] = 'SELL'
                    else:
                        prediction['recommendation'] = 'HOLD'
                    
                    return prediction
                    
                except Exception as e:
                    last_error = e
                    continue
            
            if last_error:
                raise last_error
                
        except Exception as e:
            raise Exception(f"Error generating prediction for {stock_symbol}: {str(e)}")
    
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
        context = self._prepare_context(symbol, hist, rsi)
        
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