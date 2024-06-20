import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import numpy as np
from datetime import datetime

url_list = []
date_time = []
news_text = []
headlines = []

for i in range(1, 3):
    url = 'https://oilprice.com/Energy/Crude-Oil/Page-{}.html'.format(i)
    request = requests.get(url)
    soup = BeautifulSoup(request.text, "html.parser")
    for links in soup.find_all('div', {'class': 'categoryArticle'}):
        for info in links.find_all('a'):
            if info.get('href') not in url_list:
                url_list.append(info.get('href'))

for www in url_list:
    headlines.append(www.split("/")[-1].replace('-', ' '))
    request = requests.get(www)
    soup = BeautifulSoup(request.text, "html.parser")
    for dates in soup.find_all('span', {'class': 'article_byline'}):
        dt_str = dates.text.split('-')[-1].strip().replace('CDT', '').replace('CST', '').strip()
        date_time.append(datetime.strptime(dt_str, '%b %d, %Y, %I:%M %p'))

    temp = []
    for news in soup.find_all('p'):
        temp.append(news.text)
    for last_sentence in reversed(temp):
        if last_sentence.split(" ")[0] == "By" and last_sentence.split(" ")[-1] == "Oilprice.com":
            break
        elif last_sentence.split(" ")[0] == "By":
            break
    joined_text = ' '.join(temp[temp.index("More Info") + 1:temp.index(last_sentence)])
    news_text.append(joined_text)

news_df = pd.DataFrame({'Date': date_time,
                        'Headline': headlines,
                        'News': news_text,
                        })

# Convert Date column to datetime
news_df["Date"] = pd.to_datetime(news_df["Date"])

# Download stock data
stock_data = yf.download('XOM', start=news_df["Date"].min().date(), end=news_df["Date"].max().date())
stock_data = stock_data.reset_index()

# Print date range inspection
print("News date range:", news_df["Date"].min(), news_df["Date"].max())
print("Stock data date range:", stock_data["Date"].min(), stock_data["Date"].max())

# Merge datasets
merged_df = pd.merge(news_df, stock_data, on='Date', how='inner')

# Debugging print statements
print("Before merge:", len(news_df), len(stock_data))
print("After merge:", len(merged_df))

if merged_df.empty:
    print("Merged DataFrame is empty. No matching dates between news and stock data.")
else:
    analyser = SentimentIntensityAnalyzer()

    def comp_score(text):
        return analyser.polarity_scores(text)["compound"]

    merged_df["sentiment"] = merged_df["News"].apply(comp_score)

    merged_df["Signal"] = merged_df["sentiment"].apply(lambda score: "Buy" if score > 0.1 else ("Sell" if score < -0.1 else "Hold"))

    initial_capital = 100000
    shares = 0
    capital = initial_capital
    portfolio_value = []

    for i in range(len(merged_df)):
        if merged_df.loc[i, "Signal"] == "Buy" and capital >= merged_df.loc[i, "Close"]:
            shares_to_buy = capital // merged_df.loc[i, "Close"]
            capital -= shares_to_buy * merged_df.loc[i, "Close"]
            shares += shares_to_buy
        elif merged_df.loc[i, "Signal"] == "Sell" and shares > 0:
            capital += shares * merged_df.loc[i, "Close"]
            shares = 0

        portfolio_value.append(capital + shares * merged_df.loc[i, "Close"])

    end_portfolio_value = portfolio_value[-1]
    start_portfolio_value = portfolio_value[0]
    years = len(merged_df) / 252

    cagr = (end_portfolio_value / start_portfolio_value) ** (1 / years) - 1
    av = (end_portfolio_value / start_portfolio_value) ** (1 / years)
    daily_returns = np.diff(portfolio_value) / portfolio_value[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    print(f"Compound Annual Growth Rate (CAGR): {cagr:.2%}")
    print(f"Average Annual Return (AV): {av:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
