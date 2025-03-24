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

def save_to_excel(predictions, file_path):
    """Save predictions to Excel file with timestamp."""
    # Load existing data
    existing_data = load_existing_data(file_path)
    
    # Create new data with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = []
    
    for stock, data in predictions.items():
        new_data.append({
            'Timestamp': timestamp,
            'Stock': stock,
            'Predicted Price': data['price'],
            'Confidence': f"{data['confidence']*100:.2f}%",
            'RSI': data['rsi'],
            'Time Horizon': data['time_horizon'],
            'Reasoning': data['reasoning']
        })
    
    # Convert new data to DataFrame
    new_df = pd.DataFrame(new_data)
    
    # Combine with existing data
    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
    
    # Save to Excel with formatting
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        combined_df.to_excel(writer, index=False, sheet_name='Stock Predictions')
        
        # Get the workbook and the worksheet
        workbook = writer.book
        worksheet = writer.sheets['Stock Predictions']
        
        # Format column widths
        for idx, col in enumerate(combined_df.columns):
            max_length = max(
                combined_df[col].astype(str).apply(len).max(),
                len(str(col))
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 100)
        
        # Format timestamp column
        for cell in worksheet['A'][1:]:
            cell.number_format = 'yyyy-mm-dd hh:mm:ss'

def main():
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('--stocks', nargs='+', required=True, help='List of stock symbols to analyze')
    parser.add_argument('--output', default='results.json', help='Output JSON file path')
    args = parser.parse_args()

    agent = StockAgent()
    predictions = {}

    print("\n=== Stock Analysis Results ===\n")
    print("Predictions:\n")

    for stock in args.stocks:
        try:
            prediction = agent.predict(stock)
            predictions[stock] = prediction
            
            print(f"{stock}:")
            print(f"  Predicted Price: ${prediction['price']:.2f}")
            print(f"  Confidence: {prediction['confidence']*100:.2f}%")
            print(f"  RSI: {prediction['rsi']:.2f}")
            print(f"  Time Horizon: {prediction['time_horizon']}")
            print(f"  Reasoning: {prediction['reasoning']}\n")
        except Exception as e:
            print(f"Error analyzing {stock}: {str(e)}\n")

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Save to Excel on desktop
    desktop_path = get_desktop_path()
    excel_file = os.path.join(desktop_path, 'stock_predictions.xlsx')
    save_to_excel(predictions, excel_file)
    print(f"\nResults saved to {args.output}")
    print(f"Results also saved to Excel file: {excel_file}")

    # Generate recommendations
    buy_stocks = [stock for stock, pred in predictions.items() if pred['confidence'] >= 0.7]
    hold_stocks = [stock for stock, pred in predictions.items() if 0.5 <= pred['confidence'] < 0.7]
    sell_stocks = [stock for stock, pred in predictions.items() if pred['confidence'] < 0.5]

    print("\nRecommendations:\n")
    if buy_stocks:
        print("BUY:")
        for stock in buy_stocks:
            print(f"  - {stock}")
    if hold_stocks:
        print("HOLD:")
        for stock in hold_stocks:
            print(f"  - {stock}")
    if sell_stocks:
        print("SELL:")
        for stock in sell_stocks:
            print(f"  - {stock}")

if __name__ == "__main__":
    main() 