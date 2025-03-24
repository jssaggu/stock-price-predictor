from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

def format_stock_data(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Format stock data into a dictionary with relevant statistics
    """
    if stock_data.empty:
        return {}
    
    latest = stock_data.iloc[-1]
    previous = stock_data.iloc[-2]
    
    return {
        'current_price': latest['Close'],
        'open_price': latest['Open'],
        'high_price': latest['High'],
        'low_price': latest['Low'],
        'volume': latest['Volume'],
        'price_change': latest['Close'] - previous['Close'],
        'price_change_percent': (latest['Close'] - previous['Close']) / previous['Close'],
        'volatility': stock_data['Close'].pct_change().std()
    }

def calculate_technical_indicators(stock_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate common technical indicators
    """
    if stock_data.empty:
        return {}
    
    # Calculate moving averages
    ma20 = stock_data['Close'].rolling(window=20).mean()
    ma50 = stock_data['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return {
        'ma20': ma20.iloc[-1],
        'ma50': ma50.iloc[-1],
        'rsi': rsi.iloc[-1],
        'price_vs_ma20': stock_data['Close'].iloc[-1] / ma20.iloc[-1] - 1,
        'price_vs_ma50': stock_data['Close'].iloc[-1] / ma50.iloc[-1] - 1
    }

def format_news_data(news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format news data into a clean structure
    """
    formatted_news = []
    for article in news_data:
        formatted_news.append({
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'url': article.get('url', ''),
            'published_at': article.get('publishedAt', ''),
            'source': article.get('source', {}).get('name', '')
        })
    return formatted_news

def validate_stock_symbol(symbol: str) -> bool:
    """
    Validate if a stock symbol is in the correct format
    """
    # Basic validation - can be enhanced based on specific requirements
    return bool(symbol and symbol.isalnum() and len(symbol) <= 5)

def get_date_range(days: int) -> tuple:
    """
    Get start and end dates for a given number of days
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d') 