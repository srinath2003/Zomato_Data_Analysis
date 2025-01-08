# Zomato Dataset Analysis and Cleaning

This project focuses on analyzing and cleaning a Zomato restaurant dataset from Bengaluru. The dataset contains detailed information about various restaurants, including ratings, cuisines, and cost for two people.

## Overview

The project involves:

1. Cleaning and preprocessing the Zomato dataset for accurate analysis.
2. Removing inconsistencies and handling missing or invalid data.
3. Performing exploratory data analysis (EDA) to uncover trends and patterns.

## Dataset

### Features

- `url`: URL of the restaurant's Zomato page.
- `name`: Name of the restaurant.
- `online_order`: Indicates if the restaurant accepts online orders.
- `book_table`: Indicates if table booking is available.
- `rate`: Average rating of the restaurant.
- `votes`: Number of votes the restaurant received.
- `phone`: Contact number of the restaurant.
- `location`: Neighborhood where the restaurant is located.
- `rest_type`: Type of restaurant.
- `cuisines`: Cuisines served by the restaurant.
- `approx_cost(for two people)`: Approximate cost for a meal for two.
- `reviews_list`: List of customer reviews with ratings.

## Steps

### Data Cleaning

- Removed rows with invalid or missing values in critical columns such as `rate` and `approx_cost(for two people)`.
- Handled entries with non-numeric values in numeric columns by converting them into appropriate formats.
- Standardized text columns like `name` and `cuisines` by converting them to lowercase.
- Extracted useful features such as `Cuisine_Count` to indicate the number of cuisines offered by a restaurant.

### Exploratory Data Analysis (EDA)

- Visualized the distribution of restaurant types, ratings, and costs.
- Analyzed trends in online ordering and table booking preferences.
- Identified top locations based on average ratings and unique hotel counts.
- Explored relationships between cost, rating, and other features.

## Visualizations

- Bar charts and histograms showing the frequency of restaurant types and locations.
- Heatmap of feature correlations to identify relationships between variables.
- Line plots and box plots for cost and rating distributions.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Plotly
- **Tools**: Jupyter Notebook, Google Colab

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/srinath2003/Zomato_Data_Analysis.git
   cd Zomato_Data_Analysis```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt```
## Results
Cleaned dataset ready for further analysis or modeling.
Insights into restaurant trends, including popular cuisines and high-rated locations.
Visualized data to help stakeholders make informed decisions.
## Future Scope
Extend the project to build a recommendation system based on customer reviews and ratings.
Deploy the cleaned dataset for real-time analytics or integration with BI tools.
# License

## MIT License

Copyright (c) [2003] [srinath2003]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

