# Stock Price and Performance Prediction: A Hybrid Model

This project focuses on creating a hybrid model that combines **numerical analysis** of historical stock prices with **sentiment analysis** of news headlines to predict stock price performance. The hybrid approach leverages the power of data-driven algorithms and market sentiment to achieve better predictive accuracy. 

---

## **Objective**

- Build a hybrid model for **stock price prediction** and **performance analysis**.
- Combine insights from **numerical analysis** of historical stock prices and **sentiment analysis** of news headlines.
- Predict stock performance for **SENSEX (S&P BSE SENSEX)**.
- Use either **R** or **Python**, or both for separate analysis and merge the findings.

---

## **Dataset Information**

1. **Historical Stock Prices**:
   - Source: [Yahoo Finance](https://finance.yahoo.com)
   - Data includes daily closing prices, opening prices, high, low, volume, and adjusted closing prices.

2. **News Headlines for Sentiment Analysis**:
   - Source: [News Dataset](https://bit.ly/36fFPI6)
   - Contains news headlines related to financial markets.

**Note**: While SENSEX is the focus, the project allows flexibility to select a different stock and a corresponding news dataset if desired.

---

## **Technology Stack**

- **Languages**: Python, R (optional for separate analysis)
- **Libraries**:
  - **Python**: pandas, NumPy, matplotlib, seaborn, sklearn, nltk, TextBlob, transformers
  - **R**: tidyverse, ggplot2, quantmod, tm, textdata
- **Tools**:
  - Jupyter Notebook for Python
  - RStudio for R
  - Pre-trained models (e.g., BERT for sentiment analysis)

---

## **Approach**

### **Phase 1: Data Collection**
- Download historical stock prices for **SENSEX** from [Yahoo Finance](https://finance.yahoo.com).
- Download news headlines dataset from [News Dataset](https://bit.ly/36fFPI6).

### **Phase 2: Data Preprocessing**
- **Stock Prices**:
  - Clean and preprocess the historical stock prices data.
  - Extract relevant features (e.g., closing price, daily returns, volatility).
- **News Headlines**:
  - Preprocess news headlines (remove stopwords, lemmatization, etc.).
  - Perform sentiment analysis using libraries like NLTK, TextBlob, or pre-trained models.

### **Phase 3: Exploratory Data Analysis (EDA)**
- Visualize historical trends, moving averages, and volatility for stock prices.
- Analyze sentiment distribution and trends over time.

### **Phase 4: Numerical Analysis**
- Apply machine learning models (e.g., Random Forest, LSTM, ARIMA) to historical stock prices for numerical prediction.
- Evaluate model performance using metrics like RMSE and R².

### **Phase 5: Sentiment Analysis**
- Perform sentiment scoring on news headlines.
- Analyze the correlation between sentiment scores and stock price movement.

### **Phase 6: Hybrid Model Development**
- Combine findings from **numerical analysis** and **sentiment analysis**.
- Use techniques such as feature engineering or weighted aggregation to integrate both datasets.
- Train a hybrid model (e.g., XGBoost, Neural Networks) on the combined dataset.

### **Phase 7: Evaluation**
- Evaluate the hybrid model using metrics like MAE, RMSE, and accuracy.
- Compare the hybrid model's performance against standalone models.

---

## **Results and Insights**
- The hybrid model captures both market trends (numerical analysis) and market sentiment (sentiment analysis).
- Results include:
  - Accuracy of stock price predictions.
  - Influence of news sentiment on stock performance.
  - Comparative analysis of standalone vs. hybrid models.

---

## **Procedure for Usage**

1. **Set Up Environment**:
   - Install required libraries: `pip install -r requirements.txt` (for Python) or use `install.packages()` (for R).
   
2. **Data Collection**:
   - Download the stock price data from Yahoo Finance.
   - Download the news dataset from the given source.

3. **Run Notebooks**:
   - Use `Stock_Numerical_Analysis.ipynb` for numerical analysis.
   - Use `News_Sentiment_Analysis.ipynb` for sentiment analysis.
   - Combine findings in `Hybrid_Model.ipynb`.

4. **Results**:
   - Visualizations, accuracy metrics, and hybrid model insights are generated in the respective notebooks.

---

## **Key Takeaways**

1. **Numerical Analysis** provides trend-based insights into stock performance.
2. **Sentiment Analysis** captures the psychological influence of market news on stock movement.
3. The **Hybrid Model** bridges the gap between numerical trends and market sentiment for improved prediction accuracy.

---

## **Future Scope**
- Incorporate real-time data feeds for dynamic stock prediction.
- Extend sentiment analysis to include social media platforms like Twitter.
- Explore additional stock indices or global markets.

---
