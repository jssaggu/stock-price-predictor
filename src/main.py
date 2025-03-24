from agent.stock_agent import StockAgent
import argparse
import json
from typing import List

def format_price(price) -> str:
    if price is None:
        return "N/A"
    return f"${price:.2f}"

def format_confidence(confidence) -> str:
    if confidence is None:
        return "N/A"
    return f"{confidence:.2%}"

def main():
    parser = argparse.ArgumentParser(description='Stock Price Predictor')
    parser.add_argument('--stocks', nargs='+', required=True, help='List of stock symbols to analyze')
    parser.add_argument('--output', default='results.json', help='Output file path for results')
    args = parser.parse_args()

    agent = StockAgent()
    predictions = agent.analyze_stocks(args.stocks)

    print("\n=== Stock Analysis Results ===\n")
    print("Predictions:\n")

    for symbol, prediction in predictions.items():
        print(f"{symbol}:")
        print(f"  Predicted Price: ${prediction['price']:.2f}")
        print(f"  Confidence: {prediction['confidence']*100:.2f}%")
        print(f"  RSI: {prediction['rsi']:.2f}")
        print(f"  Time Horizon: {prediction['time_horizon']}")
        print(f"  Reasoning: {prediction['reasoning']}\n")

    # Determine recommendations based on confidence threshold
    confidence_threshold = 0.7  # You can adjust this threshold
    buy_recommendations = []
    hold_recommendations = []
    sell_recommendations = []

    for symbol, prediction in predictions.items():
        if prediction['confidence'] >= confidence_threshold:
            if prediction['price'] > prediction.get('current_price', 0):
                buy_recommendations.append(symbol)
            else:
                sell_recommendations.append(symbol)
        else:
            hold_recommendations.append(symbol)

    print("Recommendations:\n")
    if buy_recommendations:
        print("BUY:")
        for symbol in buy_recommendations:
            print(f"  - {symbol}")
    if hold_recommendations:
        print("HOLD:")
        for symbol in hold_recommendations:
            print(f"  - {symbol}")
    if sell_recommendations:
        print("SELL:")
        for symbol in sell_recommendations:
            print(f"  - {symbol}")

    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 