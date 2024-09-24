# Exploring Social Media Reactions to Disasters Using Text Mining and Sentiment Analysis

## Project Overview
Natural disasters like hurricanes, earthquakes, and wildfires can have profound impacts on communities. Social media platforms serve as key spaces where individuals share their experiences, seek assistance, and express emotions during these crises. This project explores these social media reactions using **text mining** and **sentiment analysis**, aiming to provide insights into the collective mindset of affected communities during and after disasters.

## Purpose
The goal of this project is to analyze the emotional tone and sentiment of social media posts to:
- Understand the **public sentiment** during crises.
- Identify **real-time needs** of communities affected by natural disasters.
- Improve **disaster response strategies** and **mental health support** efforts.
- Ensure **inclusive recovery efforts** by capturing diverse voices, especially from marginalized communities.

## Key Features
- **Text Mining:** Extract meaningful information from vast amounts of social media data to understand trends and common themes.
- **Sentiment Analysis:** Classify social media posts by emotional tone (positive, negative, neutral) to gauge the public's emotional responses.
- **Real-time Insights:** Utilize social media’s immediacy to provide up-to-date information on disaster impact and responses.
- **Diversity in Responses:** Focus on the varied perspectives of survivors, responders, and external observers.

## Tools & Technologies
- **Python** for data processing and text mining.
- **Natural Language Processing (NLP)**:
  - `nltk`, `TextBlob`, `Spacy` for sentiment and language analysis.
  - `SentimentIntensityAnalyzer` and `NRCLex` for emotional tone detection.
- **Data Extraction**:
  - **Web Scraping**: `requests` and `BeautifulSoup` for retrieving data from web sources.
  - **Guardian News API** and **World News API** for news article collection.
- **Data Analysis & Machine Learning**:
  - `pandas`, `numpy` for data handling.
  - `KMeans`, `PCA`, `StandardScaler` for clustering and dimensionality reduction.
- **Data Visualization**:
  - `matplotlib`, `seaborn` for visualizations and trends.
  - `wordcloud` for generating word clouds from text data.
- **Network Analysis**: `networkx` for social network analysis and connectivity between social media mentions.

## Methodology

1. **Data Extraction**:
   - Articles related to natural disasters (e.g., the **Turkey-Syria earthquake**) are collected from **Guardian News API** and **World News API**.
   - Example of data extraction:
     ```python
     # Extracting data from Guardian News API
     data = requests.get("https://content.guardianapis.com/search?q=turkey%20syria%20earthquake&from-date=2023-02-06&page-size=100&show-fields=bodyText&api-key=YOUR_API_KEY")
     df_news = pd.DataFrame(data.json()['response']['results'])
     df_news['content'] = df_news['fields'].apply(lambda x: x['bodyText'])
     ```
   - The datasets from different sources are combined for comprehensive analysis.

2. **Data Processing**:
   - Text data is cleaned using techniques like stopword removal, tokenization, and stemming.
   - Example code:
     ```python
     stop_words = set(stopwords.words('english'))
     df_news['cleaned_content'] = df_news['content'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word not in stop_words]))
     ```

3. **Sentiment & Emotion Analysis**:
   - Sentiment scores are calculated using tools like **VADER** and **TextBlob** to classify the emotional tone of the posts.
   - The **NRCLex** library is used to detect specific emotions (e.g., anger, fear, joy).
   
4. **Clustering & Insights**:
   - Text data is clustered using **KMeans** and visualized using **PCA** to identify common themes in social media reactions.
   - Example clustering:
     ```python
     vectorizer = TfidfVectorizer(max_features=5000)
     X = vectorizer.fit_transform(df_news['cleaned_content'])
     kmeans = KMeans(n_clusters=5, random_state=42)
     df_news['cluster'] = kmeans.fit_predict(X)
     ```

5. **Visualization**:
   - Word clouds and sentiment trend graphs are generated to visualize the most common terms and the evolution of public sentiment over time.
   - Example of generating a word cloud:
     ```python
     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_news['cleaned_content']))
     plt.imshow(wordcloud, interpolation='bilinear')
     plt.axis('off')
     plt.show()
     ```

## Project Structure
```
|-- data/            # Contains datasets (raw and processed)
|-- notebooks/       # Jupyter notebooks for analysis
|-- src/             # Python scripts for data collection and preprocessing
|-- reports/         # Generated reports and visualizations
|-- README.md        # Project documentation
|-- requirements.txt # List of dependencies
```

## Installation
To get started with the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/disaster-social-media-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd disaster-social-media-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Collection:**
   - Use the provided scripts in `src/` to collect social media and news data from APIs.

2. **Preprocessing:**
   - Clean and preprocess the collected text data (stopwords removal, tokenization, stemming).

3. **Sentiment Analysis:**
   - Apply sentiment analysis using NLP libraries to categorize posts as positive, negative, or neutral.

4. **Visualization:**
   - Generate visualizations to understand trends in sentiment over time and across different demographics.

## Contributions
Contributions are welcome! If you would like to collaborate, please fork the repository and submit a pull request with a clear description of your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
