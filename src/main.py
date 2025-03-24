from agent.stock_agent import StockAgent
import argparse
import json
import os
from datetime import datetime
import pandas as pd
from typing import List

def format_price(price) -> str:
    if price is None:
        return "N/A"
    return f"${price:.2f}"

def format_confidence(confidence) -> str:
    if confidence is None:
        return "N/A"
    return f"{confidence:.2%}"

def get_desktop_path():
    """Get the path to the user's desktop."""
    return os.path.expanduser("~/Desktop")

def load_existing_data(file_path):
    """Load existing data from Excel file if it exists."""
    if os.path.exists(file_path):
        try:
            return pd.read_excel(file_path)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def save_to_excel(predictions, output_file):
    """Save predictions to Excel with separate sheets for each stock"""
    try:
        # Try to load existing Excel file
        existing_data = {}
        if os.path.exists(output_file):
            with pd.ExcelFile(output_file) as xls:
                for sheet_name in xls.sheet_names:
                    if sheet_name != 'Summary':
                        existing_data[sheet_name] = pd.read_excel(xls, sheet_name)
    except Exception as e:
        print(f"Warning: Could not load existing Excel file: {str(e)}")
        existing_data = {}

    # Create a new Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Create a summary sheet with all stocks
        summary_data = []
        for stock, data in predictions.items():
            if isinstance(data, dict):
                summary_data.append({
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Stock': stock,
                    'Predicted Price': data.get('price', 0),
                    'Confidence': data.get('confidence', 0),
                    'RSI': data.get('rsi', 0),
                    'Time Horizon': data.get('time_horizon', 'N/A'),
                    'Reasoning': data.get('reasoning', 'Error occurred'),
                    'Recommendation': data.get('recommendation', 'ERROR')
                })
        
        # Save summary sheet
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create individual sheets for each stock
        for stock, data in predictions.items():
            if isinstance(data, dict):
                # Create new data for this stock
                new_data = pd.DataFrame([{
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Predicted Price': data.get('price', 0),
                    'Confidence': data.get('confidence', 0),
                    'RSI': data.get('rsi', 0),
                    'Time Horizon': data.get('time_horizon', 'N/A'),
                    'Reasoning': data.get('reasoning', 'Error occurred'),
                    'Recommendation': data.get('recommendation', 'ERROR')
                }])
                
                # Combine with existing data if available
                if stock in existing_data:
                    combined_data = pd.concat([existing_data[stock], new_data], ignore_index=True)
                    combined_data.to_excel(writer, sheet_name=stock, index=False)
                else:
                    new_data.to_excel(writer, sheet_name=stock, index=False)

def main():
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('--stocks', nargs='+', required=True, help='List of stock symbols to analyze')
    parser.add_argument('--output', default='results.json', help='Output file path for results')
    args = parser.parse_args()

    # Initialize the stock agent
    agent = StockAgent()

    # Analyze each stock
    predictions = {}
    print("\n=== Stock Analysis Results ===\n")
    print("Predictions:\n")

    for stock in args.stocks:
        try:
            prediction = agent.predict(stock)
            predictions[stock] = prediction
            print(f"{stock}:")
            print(f"  Predicted Price: ${prediction['price']:.2f}")
            print(f"  Confidence: {prediction['confidence']:.2f}%")
            print(f"  RSI: {prediction['rsi']:.2f}")
            print(f"  Time Horizon: {prediction['time_horizon']}")
            print(f"  Reasoning: {prediction['reasoning']}\n")
        except Exception as e:
            print(f"Error analyzing {stock}: {str(e)}\n")
            predictions[stock] = {
                'price': 0,
                'confidence': 0,
                'rsi': 0,
                'time_horizon': 'N/A',
                'reasoning': f'Error: {str(e)}',
                'recommendation': 'ERROR'
            }

    # Save results to JSON
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save results to Excel
    excel_file = os.path.expanduser('~/OneDrive/stock_predictions.xlsx')
    save_to_excel(predictions, excel_file)
    print(f"Results also saved to Excel file: {excel_file}")

    # Print recommendations
    print("\nRecommendations:\n")
    recommendations = {
        'BUY': [],
        'HOLD': [],
        'SELL': [],
        'ERROR': []
    }
    
    for stock, data in predictions.items():
        if isinstance(data, dict):
            recommendations[data.get('recommendation', 'ERROR')].append(stock)
    
    for action, stocks in recommendations.items():
        if stocks and action != 'ERROR':
            print(f"{action}:")
            for stock in stocks:
                print(f"  - {stock}")

if __name__ == "__main__":
    main() 