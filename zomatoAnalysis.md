<h2><center>ANALYSIS ZOMATO DATASET</center></h2>

### ABOUT DATASET

   - <b> url </b>:  contains the url of the restaurant in the zomato website

   - <b> address </b> :  contains the address of the restaurant in Bengaluru

   - <b> name </b>:  contains the name of the restaurant

   - <b> online_order </b> :  whether online ordering is available in the restaurant or not

   - <b> book_table</b> :  table book option available or not

   - <b> rate </b> :  contains the overall rating of the restaurant out of 5

   - <b> votes </b> :  contains total number of rating for the restaurant as of the above mentioned date

   - <b> phone </b> :  contains the phone number of the restaurant

   - <b> location </b> :  contains the neighborhood in which the restaurant is located

   - <b> rest_type </b> :  restaurant type

   - <b> dish_liked </b> :  dishes people liked in the restaurant

   - <b> cuisines </b> :  food styles, separated by comma

   - <b> approx_cost(for two people) </b> :  contains the approximate cost for meal for two people

   - <b> reviews_list </b> :  list of tuples containing reviews for the restaurant, each tuple consists of two values, rating and review by the customer

   - <b> menu_item </b> :  contains list of menus available in the restaurant

   - <b> listed_in(type) </b> :  type of meal

   - <b> listed_in(city) </b> :  contains the neighborhood in which the restaurant is listed

### RECOMMANDATION SYSTEM


```python
# !pip install nltk

```


```python
# from google.colab import drive
# drive.mount('/content/drive')
```


```python
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import re
import ast
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
zomato_real = pd.read_csv("zomato.csv")

```

# Understanding Data Set



```python
zomato_real.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>rate</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>approx_cost(for two people)</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st Main Road, 2nd Stage, Banashankari, ...</td>
      <td>Jalsa</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>4.1/5</td>
      <td>775</td>
      <td>080 42297555\r\n+91 9743772233</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Pasta, Lunch Buffet, Masala Papad, Paneer Laja...</td>
      <td>North Indian, Mughlai, Chinese</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  A beautiful place to ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...</td>
      <td>Spice Elephant</td>
      <td>Yes</td>
      <td>No</td>
      <td>4.1/5</td>
      <td>787</td>
      <td>080 41714161</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Momos, Lunch Buffet, Chocolate Nirvana, Thai G...</td>
      <td>Chinese, North Indian, Thai</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  Had been here for din...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/SanchurroBangalore?cont...</td>
      <td>1112, Next to KIMS Medical College, 17th Cross...</td>
      <td>San Churro Cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.8/5</td>
      <td>918</td>
      <td>+91 9663487993</td>
      <td>Banashankari</td>
      <td>Cafe, Casual Dining</td>
      <td>Churros, Cannelloni, Minestrone Soup, Hot Choc...</td>
      <td>Cafe, Mexican, Italian</td>
      <td>800</td>
      <td>[('Rated 3.0', "RATED\n  Ambience is not that ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st Floor, Annakuteera, 3rd Stage, Banashankar...</td>
      <td>Addhuri Udupi Bhojana</td>
      <td>No</td>
      <td>No</td>
      <td>3.7/5</td>
      <td>88</td>
      <td>+91 9620009302</td>
      <td>Banashankari</td>
      <td>Quick Bites</td>
      <td>Masala Dosa</td>
      <td>South Indian, North Indian</td>
      <td>300</td>
      <td>[('Rated 4.0', "RATED\n  Great food and proper...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd Floor, Lakshmi Associates, Gandhi Baza...</td>
      <td>Grand Village</td>
      <td>No</td>
      <td>No</td>
      <td>3.8/5</td>
      <td>166</td>
      <td>+91 8026612447\r\n+91 9901210005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Panipuri, Gol Gappe</td>
      <td>North Indian, Rajasthani</td>
      <td>600</td>
      <td>[('Rated 4.0', 'RATED\n  Very good restaurant ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
  </tbody>
</table>
</div>




```python
zomato_real['url'][0]
```




    'https://www.zomato.com/bangalore/jalsa-banashankari?context=eyJzZSI6eyJlIjpbNTg2OTQsIjE4Mzc1NDc0IiwiNTkwOTAiLCIxODM4Mjk0NCIsIjE4MjI0Njc2IiwiNTkyODkiLCIxODM3MzM4NiJdLCJ0IjoiUmVzdGF1cmFudHMgaW4gQmFuYXNoYW5rYXJpIHNlcnZpbmcgQnVmZmV0In19'




```python
df = zomato_real
```


```python
name = list(df.name.unique())
```


```python
df.name.nunique()
```




    8792




```python
df.columns
```




    Index(['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes',
           'phone', 'location', 'rest_type', 'dish_liked', 'cuisines',
           'approx_cost(for two people)', 'reviews_list', 'menu_item',
           'listed_in(type)', 'listed_in(city)'],
          dtype='object')




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>rate</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>approx_cost(for two people)</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st Main Road, 2nd Stage, Banashankari, ...</td>
      <td>Jalsa</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>4.1/5</td>
      <td>775</td>
      <td>080 42297555\r\n+91 9743772233</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Pasta, Lunch Buffet, Masala Papad, Paneer Laja...</td>
      <td>North Indian, Mughlai, Chinese</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  A beautiful place to ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...</td>
      <td>Spice Elephant</td>
      <td>Yes</td>
      <td>No</td>
      <td>4.1/5</td>
      <td>787</td>
      <td>080 41714161</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Momos, Lunch Buffet, Chocolate Nirvana, Thai G...</td>
      <td>Chinese, North Indian, Thai</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  Had been here for din...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/SanchurroBangalore?cont...</td>
      <td>1112, Next to KIMS Medical College, 17th Cross...</td>
      <td>San Churro Cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.8/5</td>
      <td>918</td>
      <td>+91 9663487993</td>
      <td>Banashankari</td>
      <td>Cafe, Casual Dining</td>
      <td>Churros, Cannelloni, Minestrone Soup, Hot Choc...</td>
      <td>Cafe, Mexican, Italian</td>
      <td>800</td>
      <td>[('Rated 3.0', "RATED\n  Ambience is not that ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st Floor, Annakuteera, 3rd Stage, Banashankar...</td>
      <td>Addhuri Udupi Bhojana</td>
      <td>No</td>
      <td>No</td>
      <td>3.7/5</td>
      <td>88</td>
      <td>+91 9620009302</td>
      <td>Banashankari</td>
      <td>Quick Bites</td>
      <td>Masala Dosa</td>
      <td>South Indian, North Indian</td>
      <td>300</td>
      <td>[('Rated 4.0', "RATED\n  Great food and proper...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd Floor, Lakshmi Associates, Gandhi Baza...</td>
      <td>Grand Village</td>
      <td>No</td>
      <td>No</td>
      <td>3.8/5</td>
      <td>166</td>
      <td>+91 8026612447\r\n+91 9901210005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Panipuri, Gol Gappe</td>
      <td>North Indian, Rajasthani</td>
      <td>600</td>
      <td>[('Rated 4.0', 'RATED\n  Very good restaurant ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>Four Points by Sheraton Bengaluru, 43/3, White...</td>
      <td>Best Brews - Four Points by Sheraton Bengaluru...</td>
      <td>No</td>
      <td>No</td>
      <td>3.6 /5</td>
      <td>27</td>
      <td>080 40301477</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>Continental</td>
      <td>1,500</td>
      <td>[('Rated 5.0', "RATED\n  Food and service are ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51713</th>
      <td>https://www.zomato.com/bangalore/vinod-bar-and...</td>
      <td>Number 10, Garudachar Palya, Mahadevapura, Whi...</td>
      <td>Vinod Bar And Restaurant</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>0</td>
      <td>+91 8197675843</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>Finger Food</td>
      <td>600</td>
      <td>[]</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51714</th>
      <td>https://www.zomato.com/bangalore/plunge-sherat...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>Plunge - Sheraton Grand Bengaluru Whitefield H...</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>Finger Food</td>
      <td>2,000</td>
      <td>[]</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>Chime - Sheraton Grand Bengaluru Whitefield Ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>4.3 /5</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>Finger Food</td>
      <td>2,500</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>ITPL Main Road, KIADB Export Promotion Industr...</td>
      <td>The Nest - The Den Bengaluru</td>
      <td>No</td>
      <td>No</td>
      <td>3.4 /5</td>
      <td>13</td>
      <td>+91 8071117272</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar, Casual Dining</td>
      <td>NaN</td>
      <td>Finger Food, North Indian, Continental</td>
      <td>1,500</td>
      <td>[('Rated 5.0', 'RATED\n  Great ambience , look...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
  </tbody>
</table>
<p>51717 rows × 17 columns</p>
</div>




```python
new_shop =df[df["rate"] == "NEW" ]
```


```python
(new_shop.shape)[0]
```




    2208



# Cleaning Data Set



```python
# Remove rows where 'column_name' contains the substring 'new'
df = df[~df['rate'].str.contains('NEW', case=False, na=False)]

```


```python
# Remove rows where 'rate' contains 'new' or is None
df = df[~df['rate'].str.contains('new', case=False, na=False) & df['rate'].notna()]

```


```python
# Remove rows where 'column_name' contains the substring 'new'
df = df[~df['rate'].str.contains('-', case=False, na=False)]

```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>rate</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>approx_cost(for two people)</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st Main Road, 2nd Stage, Banashankari, ...</td>
      <td>Jalsa</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>4.1/5</td>
      <td>775</td>
      <td>080 42297555\r\n+91 9743772233</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Pasta, Lunch Buffet, Masala Papad, Paneer Laja...</td>
      <td>North Indian, Mughlai, Chinese</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  A beautiful place to ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...</td>
      <td>Spice Elephant</td>
      <td>Yes</td>
      <td>No</td>
      <td>4.1/5</td>
      <td>787</td>
      <td>080 41714161</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Momos, Lunch Buffet, Chocolate Nirvana, Thai G...</td>
      <td>Chinese, North Indian, Thai</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  Had been here for din...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/SanchurroBangalore?cont...</td>
      <td>1112, Next to KIMS Medical College, 17th Cross...</td>
      <td>San Churro Cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>3.8/5</td>
      <td>918</td>
      <td>+91 9663487993</td>
      <td>Banashankari</td>
      <td>Cafe, Casual Dining</td>
      <td>Churros, Cannelloni, Minestrone Soup, Hot Choc...</td>
      <td>Cafe, Mexican, Italian</td>
      <td>800</td>
      <td>[('Rated 3.0', "RATED\n  Ambience is not that ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st Floor, Annakuteera, 3rd Stage, Banashankar...</td>
      <td>Addhuri Udupi Bhojana</td>
      <td>No</td>
      <td>No</td>
      <td>3.7/5</td>
      <td>88</td>
      <td>+91 9620009302</td>
      <td>Banashankari</td>
      <td>Quick Bites</td>
      <td>Masala Dosa</td>
      <td>South Indian, North Indian</td>
      <td>300</td>
      <td>[('Rated 4.0', "RATED\n  Great food and proper...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd Floor, Lakshmi Associates, Gandhi Baza...</td>
      <td>Grand Village</td>
      <td>No</td>
      <td>No</td>
      <td>3.8/5</td>
      <td>166</td>
      <td>+91 8026612447\r\n+91 9901210005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Panipuri, Gol Gappe</td>
      <td>North Indian, Rajasthani</td>
      <td>600</td>
      <td>[('Rated 4.0', 'RATED\n  Very good restaurant ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51709</th>
      <td>https://www.zomato.com/bangalore/the-farm-hous...</td>
      <td>136, SAP Labs India, KIADB Export Promotion In...</td>
      <td>The Farm House Bar n Grill</td>
      <td>No</td>
      <td>No</td>
      <td>3.7 /5</td>
      <td>34</td>
      <td>+91 9980121279\n+91 9900240646</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>NaN</td>
      <td>North Indian, Continental</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  Ambience- Big and spa...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51711</th>
      <td>https://www.zomato.com/bangalore/bhagini-2-whi...</td>
      <td>139/C1, Next To GR Tech Park, Pattandur Agraha...</td>
      <td>Bhagini</td>
      <td>No</td>
      <td>No</td>
      <td>2.5 /5</td>
      <td>81</td>
      <td>080 65951222</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>Biryani, Andhra Meal</td>
      <td>Andhra, South Indian, Chinese, North Indian</td>
      <td>800</td>
      <td>[('Rated 4.0', 'RATED\n  A fine place to chill...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>Four Points by Sheraton Bengaluru, 43/3, White...</td>
      <td>Best Brews - Four Points by Sheraton Bengaluru...</td>
      <td>No</td>
      <td>No</td>
      <td>3.6 /5</td>
      <td>27</td>
      <td>080 40301477</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>Continental</td>
      <td>1,500</td>
      <td>[('Rated 5.0', "RATED\n  Food and service are ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>Chime - Sheraton Grand Bengaluru Whitefield Ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>4.3 /5</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>Finger Food</td>
      <td>2,500</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>ITPL Main Road, KIADB Export Promotion Industr...</td>
      <td>The Nest - The Den Bengaluru</td>
      <td>No</td>
      <td>No</td>
      <td>3.4 /5</td>
      <td>13</td>
      <td>+91 8071117272</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar, Casual Dining</td>
      <td>NaN</td>
      <td>Finger Food, North Indian, Continental</td>
      <td>1,500</td>
      <td>[('Rated 5.0', 'RATED\n  Great ambience , look...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
    </tr>
  </tbody>
</table>
<p>41665 rows × 17 columns</p>
</div>




```python
df["menu_item"].unique()
```




    array(['[]',
           "['Chocolate Fantasy (Pack Of 5)', 'Pan Cake (Pack Of 6)', 'Gulab Jamun (Pack Of 10)', 'Gulkand Shot (Pack Of 5)', 'Chocolate Decadence (Pack of 2)', 'CheeseCake (Pack Of 2)', 'Red Velvet Slice Cake (Pack of 2)', 'Red Velvet Slice Cake & Cheese Cake (Pack of 2)', 'Red Velvet Slice Cake & Chocolate Decadence Cake (Pack of 2)', 'Hazelnut Brownie (Pack of 2)', 'Moments', 'Red Velvet Cake With Butter Cream Frosting (750 Gm)', 'Red Velvet Slice Cake (Pack of 2)', 'Red Velvet Slice Cake & Cheese Cake (Pack of 2)', 'Red Velvet Slice Cake & Chocolate Decadence Cake (Pack of 2)', 'Red Velvet Slice Cake (Pack of 1)', 'Valentine Red Velvet Jar', 'Valentine Chocolate Jar', 'Valentines Jar Combo', 'Pink Guava 500 ML', 'Oreo Vanilla 500 ML', 'Cookie Crumble 500 ML', 'Chocolate Fantasy', 'Gulkand-E-Bahar', 'Pan Cake', 'Hazelnut Brownie (Pack Of 1)', 'Gulab Jamun (Pack Of 2)', 'Plum Cake', 'Red Velvet Cake With Butter Cream Frosting (750 Gm)', 'Chocolate Mud Cake (700 Gms)', 'CheeseCake (Pack of 1)', 'Chocolate Decadence (Pack of 1)', 'Red Velvet Slice Cake (Pack of 1)']",
           "['Chole Kulcha Meal', 'Upvas Aloo Paratha With Dahi', 'Singhada Aloo Paratha with Hare Tamatar Ki Sabji', 'Smoked Butter Chicken Combo', 'Paneer Methi Chaman Combo', 'Mutton Bhuna Combo', 'Rajma Masala Meal', 'Dal Makhani Veg Starter Combo', 'Dal Makhani Non-Veg Starter Combo', 'Malai Kofta Combo', 'Jumbo Chicken Wrap', 'Jumbo Veg Wrap', 'Jumbo Falafel Salsa Wrap', 'Chicken Overload Jumbo Wrap', 'Veg Pizza Wrap', 'Chicken Pizza Wrap', 'Mexican Potato Salsa Wrap', 'American Smokey Sausage Wrap', 'Makhani Falafel Wrap', 'Mutton Overload Wrap', 'Mac & Cheese Chicken Wrap', 'Mac & Cheese Veg Wrap', 'Barbeque Chicken Wrap', 'Mutton Boti Wrap', 'Masala Paneer Tikka Wrap', 'Fiery Paneer Tikka Wrap', 'Masala Chicken Tikka Wrap', 'Cheesy Corn Salsa Wrap', 'Chicken Mayo Wrap', 'Cheese Melt Chicken Wrap', 'Cheese Melt Paneer Wrap', 'Double Cheese Meatball Wrap', 'Reshmi Chicken Kebab Wrap', 'Egg Cheese Sausage Wrap', 'Double Egg Chatpata Wrap', 'Cheesy Potato Wrap', 'Veg Falafel Wrap', 'Chicken Bhuna Wrap', 'Chatpate Chole Wrap', 'Fiery Paneer Wrap', 'Fiery Chicken Wrap', 'Mac & Cheese Wrap', 'Hare Tamatar & Sabudana Wada Royal Thali', 'Singhada Aloo Paratha & Hare Tamatar Royal Satvik Meal', 'Smoked Butter Chicken With Omelette', 'Rajma Masala Royal Combo', 'Paneer Methi Chaman Royal Combo', 'Mutton Bhuna Royal Combo', 'Smoked Butter Chicken Royal Combo', 'Malai Kofta Royal Combo', 'Fusion Breakfast', 'Pan Cake', 'Aloo Paratha Combo', 'Chai for 4', 'Cheesy Chicken Meatballs', 'Peach Tea (Serves 4)', 'Falafel Nuggets with Mayo Dip', 'Potato Chilli Shots with Mayo Dip', 'Pan Cake', 'Kashmiri Kahwa (Serves 4)', 'Masala Chai (Serves 4)', 'Chai for 4 + Nature valley bar', 'Kulcha', 'Flavorful Rice Tub', 'Dal Makhani Bowl (Half KG)', 'Smoked Butter Chicken Bowl (Half KG)', 'Mutton Bhuna Bowl (Half KG)', 'Singhada Aloo Paratha Tub', 'Curd Bowl (Half KG)', 'Hare Tamatar Ki Subji (Half KG)', 'Rajma Masala Bowl (Half KG)', 'Triangle Paratha Tub', 'Malai Kofta Bowl (Half Kg)', 'Chocolate Fantasy', 'SWIG Jeera Masala', 'SWIG Green apple', 'Kesar Muesli', 'Gulab Jamun (Pack Of 2)', 'Plum Cake', 'Gulab Jamun (Pack of 1)', 'Moments', 'Hazelnut Brownie (Pack Of 1)', 'CheeseCake (Pack of 1)', 'Chocolate Decadence (Pack of 1)', 'Red Velvet Slice Cake (Pack of 1)', 'Mint Chaas']",
           ...,
           '[\'Veg Thai Green Curry\', \'BBQ Sloppy Chicken Burger\', "Harry\'s Farm House Pizza", \'Egg Schezwan Fried Rice\', \'Chicken Schezwan Noodles\', \'Veg Platter\', \'Non Veg Platter\', \'Tom Yum Goong\', \'Tandoori Kalonji Aloo\', \'Tandoori Pesto Flowerets\', \'Paneer and Pineapple Tikka\', \'Adraki Soy Chilli Chicken Kebab\', \'Tandoori Chicken Lollipop\', \'Parpika Choosa\', \'Lamb Seekh Kebab\', \'Achari Fish Tikka\', \'Rumali Roti\', \'Naan\', \'Butter Naan\', \'Garlic Naan\', \'Butter Garlic Naan\', \'Cheese Naan\', \'Butter Garlic Cheese Naan\', \'Garlic Cheese Naan\', \'Veg Fried Rice\', \'Veg Schezwan Fried Rice\', \'Veg Hakka Noodles\', \'Veg Schezwan Noodles\', \'Veg Singapore Noodles\', \'Egg Fried Rice\', \'Egg Schezwan Fried Rice\', \'Egg Hakka Noodles\', \'Egg Schezwan Noodles\', \'Egg Singapore Noodles\', \'Chicken Hakka Noodles\', \'Chicken Schezwan Noodles\', \'Chicken Singapore Noodles\', \'Prawns Fried Rice\', \'Prawns Schezwan Fried Rice\', \'Prawns Hakka Noodles\', \'Prawns Schezwan Noodles\', \'Prawns Singapore Noodles\', \'Margherita Pizza\', "Harry\'s Farm House Pizza", \'Classique Paneer Tikka Pizza\', \'Veg Exotica Pizza\', \'Thai Basil Chicken Pizza\', \'Classic Chicken Tikka Pizza\', \'Spicy Pepperoni Pizza\', \'Penne Rustica\', \'Spaghetti Aglio e Olio Pasta\', \'HarryÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x83Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82Â¢Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x80Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x99s Mac and Cheese Pasta\', \'Penne Vodkatini\', \'Baked Butter Chicken Lasaqna\', \'Spaghetti Carbonara\', \'Straight Up Veg Burger\', \'BBQ Sloppy Chicken Burger\', \'HarryÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x83Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82Â¢Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x80Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x99s Lamb Burger\', \'Jazz Burger\', \'Paneer Bhurji Pav\', \'Egg Kejriwal\', \'Anda Bhurji Pav\', \'Omelette Pav\', \'Egg Chilly\', \'Goan Rassa Omelette with Pav\', \'Veg Wok Tossed Ginger Sprout Noodles with Crackling Spinach\', \'Veg Kung Pao Curry\', \'Veg Thai Green Curry\', \'Veg Laksa\', \'Chicken Wok Tossed Ginger Sprout Noodles with Crackling Spinach\', \'Grilled Chilly Chicken\', \'Chicken Kung Pao Curry\', \'Chicken Thai Green Curry\', \'Chicken Laksa\', \'Sri Lankan Black Pepper Chicken\', \'Nasi Stir Fry Rice\', \'Fish Wok Tossed Ginger Sprout Noodles with Crackling Spinach\', \'Seafood Kung Pao Curry\', \'Seafood Thai Green Curry\', \'Seafood Laksa\', \'Bangkok Seafood Bowl\', \'Crispy Garlic Bread\', \'Classic Masala Peanut\', \'Salt and Pepper Corn\', \'Bombay Spiced Wedges\', \'Harrys Onion Rings\', \'French Fries\', \'Crispy Chilli Potatoes\', \'Baked Sweet Potato Wedges\', \'Peri Peri Salsa Fries\', \'Parmigiana Truffle Fries\', \'Chicken Tikka Poutine\', \'Lamb Lovers Fries\', \'Paneer Bhurji Pav\', \'Cheese Chilli Toast\', \'Pita Chips with Classic Hummus\', \'Paneer Chilli Dry\', \'Veg Smoked Quesadillas\', \'Fondue Stuffed Mushrooms\', \'Wonton Kachories\', \'Cottage Cheese Malaysian Satay\', \'Muchos Nachos Veggie\', "Harry\'s Cheese Fondue Veg Dunks", \'Non Veg Smoked Quesadillas\', \'Chilli Chicken\', \'Chicken Malaysian Satay\', \'Drunken Chicken\', \'Crispy Chicken\', \'Andhra Chicken Pepper Fry\', \'Singapore Chicken Lollipop\', \'Fiery Schezwan Pepper Chicken\', \'Chilli Chicken Roll\', \'BBQ Rum Flambeed Wings\', \'Grilled Tenderloin Chilli Fry\', \'Keema Pav\', \'Pigs in a Blanket\', \'Kerala Buff Chilly\', \'Buffalo Chicken Winglets\', \'Pattaya Beach Fish Goujons\', \'Fishermans Basket\', "Harry\'s Cheese Fondue Non Veg Dunks", \'Paneer Urban Tikka Masala\', "Veg Harry\'s Stroganoff", \'Chicken Urban Tikka Masala\', \'Jamaican Jerk Chicken\', "Chicken Harry\'s Stroganoff", \'Harrys Pepper Chicken\', \'Peppered Tenderloin Steak\', "Harry\'s Beer Batter Fish", \'Smoked Paprika Fish\', \'Darsaan\', \'Sizzling Brownie\']',
           '[\'Chicken Wings\', \'Pepper Chicken\', "Beef Grilled One\'s Burger", \'Spicy Louisiana Pizza\', \'Chilly Chicken Pizza\', \'Chicken Mexicana Pizza\', \'Veg Italian Tomato and Fresh Basil Soup\', \'Chicken Lemon Coriander Soup\', \'Southern Fried Chicken\', \'Chicken Wings\', \'Gambas\', \'Bbq Jerk Chicken\', \'Chicken Nachos Carnitos\', \'Spicy Chicken Carrebbean Sausages\', \'Spicy Pork Carrebbean Sausages\', \'Pork Nachos Carnitos\', \'Beef Spicy New Orleans Braised Meat\', \'Cajun Spiced Fish Fingers\', \'Pepper Chicken\', \'Chicken Apollo Style\', \'Prawns Apollo Style\', \'Chicken Streetside Chilly Style\', \'Beef Streetside Chilly Style\', \'Pork Streetside Chilly Style\', \'Chicken Manchurian Style\', "Chicken Spicy Chef\'s Special Style", "Beef Spicy Chef\'s Special Style", "Chicken Kim\'s Style", \'Beef Nachos Carnitos\', "Beef Kim\'s Style", "Pork Kim\'s Style", \'Chettinad Chicken Roast\', \'Chettinad Pudhina Chicken Roast\', \'Chicken Kabab\', "Prawns Spicy Chef\'s Special Style", \'Prawns Manchurian Style\', \'Fried Beef\', \'Pork Spicy New Orleans Braised Meat\', \'Fried Pork\', "Prawns Kim\'s Style", \'Fish Manchurian Style\', "Fish Spicy Chef\'s Special Style", \'Fish Streetside Chilly Style\', \'Fish Apollo Style\', "Fish Kim\'s Style", \'Pork Streetside Chilly Style\', "Pork Spicy Chef\'s Special Style", \'Pork Manchurian Style\', \'Beef Manchurian Style\', "Paneer Korean Kim\'s Style", \'Paneer Tangy Spinach Sauce\', \'Paneer Apollo Style Sauce\', \'Paneer Singaporean Style Sauce\', \'Paneer Ginger Chilly Garlic Sauce\', \'Paneer Ginger Chilly Garlic Sauce\', \'Paneer Ginger Chilly Garlic Sauce\', \'Paneer Hot Garlic Sauce\', \'Paneer Lemon Coriander Sauce\', "Veg Korean Kim\'s Style", \'Veg Tangy Spinach Sauce\', \'Veg Apollo Style Sauce\', "Veg Korean Kim\'s Style", \'Veg Singaporean Style Sauce\', \'Veg Ginger Chilly Garlic Sauce Burger\', \'Veg Chefs Super Spicy Style\', \'Veg Hot Garlic Sauce\', \'Lemon Coriander Sauce\', "Chicken Korean Kim\'s Style", \'Chicken Tangy Spinach Sauce\', \'Apollo Style Sauce\', \'Chicken Singaporean Style Sauce\', \'Chicken Singaporean Style Sauce\', \'Chicken Chefs Super Spicy Style\', \'Chicken Hot Garlic Sauce\', \'Chicken Hot Garlic Sauce\', "Beef Korean Kim\'s Style", \'Beef Tangy Spinach Sauce\', \'Beef Apollo Style Sauce\', \'Beef Singaporean Style Sauce\', \'Beef Ginger Chilly Garlic Sauce\', \'Beef Chefs Super Spicy Style\', \'Beef Chefs Super Spicy Style\', \'Beef Hot Garlic Sauce\', \'Beef Lemon Coriander Sauce\', "Pork Korean Kim\'s Style", \'Pork Tangy Spinach Sauce\', \'Pork Apollo Style Sauce\', \'Pork Singaporean Style Sauce\', \'Pork Ginger Chilly Garlic Sauce\', \'Pork Chefs Super Spicy Style\', \'Pork Hot Garlic Sauce\', \'Pork Lemon Coriander Sauce\', "Fish Korean Kim\'s Style", \'Fish Tangy Spinach Sauce\', \'Fish Apollo Style Sauce\', \'Fish Singaporean Style Sauce\', \'Fish Ginger Chilly Garlic Sauce\', \'Fish Chefs Super Spicy Style\', \'Fish Hot Garlic Sauce\', \'Fish Lemon Coriander Sauce\', "Korean Kim\'s Style", \'Prawns Tangy Spinach Sauce\', \'Prawns Apollo Style Sauce\', \'Prawns Apollo Style Sauce\', \'Prawns Ginger Chilly Garlic Sauce\', \'Prawns Chefs Super Spicy Style\', \'Prawns Hot Garlic Sauce\', \'Prawns Lemon Coriander Sauce\', \'Veg Stir Fried Rice\', \'Veg Mongolian Rice\', \'Veg Olivers Spicy Rice\', \'Veg Singaporean Rice\', \'Egg Stir Fried Rice\', \'Egg Mongolian Rice\', \'Egg Olivers Spicy Rice\', \'Egg Singaporean Rice\', \'Chicken Stir Fried Rice\', \'Chicken Mongolian Rice\', \'Chicken Olivers Spicy Rice\', \'Chicken Singaporean Rice\', \'Beef Stir Fried Rice\', \'Beef Mongolian Rice\', \'Beef Olivers Spicy Rice\', \'Beef Singaporean Rice\', \'Pork Stir Fried Rice\', \'Pork Mongolian Rice\', \'Pork Olivers Spicy Rice\', \'Pork Singaporean Rice\', \'Fish Stir Fried Rice\', \'Fish Mongolian Rice\', \'Fish Olivers Spicy Rice\', \'Fish Singaporean Rice\', \'Prawns Stir Fried Rice\', \'Prawns Mongolian Rice\', \'Prawns Olivers Spicy Rice\', \'Prawns Singaporean Rice\', \'Veg Stir Fried Noodles\', \'Veg Mongolian Noodles\', \'Veg Olivers Spicy Noodles\', \'Veg Singaporean Noodles\', \'Egg Stir Fried Noodles\', \'Egg Mongolian Noodles\', \'Egg Olivers Spicy Noodles\', \'Egg Singaporean Noodles\', \'Chicken Stir Fried Noodles\', \'Chicken Mongolian Noodles\', \'Chicken Olivers Spicy Noodles\', \'Chicken Singaporean Noodles\', \'Beef Stir Fried Noodles\', \'Beef Mongolian Noodles\', \'Beef Olivers Spicy Noodles\', \'Beef Singaporean Noodles\', \'Pork Stir Fried Noodles\', \'Pork Mongolian Noodles\', \'Pork Olivers Spicy Noodles\', \'Pork Singaporean Noodles\', \'Fish Stir Fried Noodles\', \'Fish Mongolian Noodles\', \'Fish Olivers Spicy Noodles\', \'Fish Singaporean Noodles\', \'Prawns Stir Fried Noodles\', \'Prawns Mongolian Noodles\', \'Prawns Olivers Spicy Noodles\', \'Prawns Singaporean Noodles\', \'Mexican Bean Burger\', \'Bean Bbq\', \'Ginger Chilly Garlic Sauce Burger\', \'Cottage Cheese Jerky Burgers\', \'Spinach Ricotta Smashers\', \'Grilled Pepper Smashers\', \'Grilled Veg Sandwich\', \'Falafel\', \'Eggplant Parm\', "Chicken Grilled One\'s Burger", "Beef Grilled One\'s Burger", \'Almost Famous Moo Burger\', \'Miami Chicken Burger\', \'Miami Beef Burger\', \'Philly Cheesesteak Burger\', \'Santa Fe Beef Burger\', \'Cleveland Sandwich\', \'Philly Cheese Steak Sandwich\', \'Cuban Sandwich\', \'Pulled Chicken Sandwich\', \'Beef Saigon Bahn Mi Sandwich\', \'Pork Saigon Bahn Mi Sandwich\', \'Cheese Garlic Bread\', \'Jalapeno Poppers\', \'Cajun Spiced Potato Wedges\', \'Bruschetta\', \'Spring Rolls\', \'Crispy Potato Fingers\', \'Manchurian Style\', \'Streeside Chilly Style\', \'Crispy Fried Babycorn\', \'American Corn Bombs\', \'Cripsy Fried Paneer\', \'Pepper Salt\', \'OliverÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x83Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82Â¢Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x80Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x99s Pepper Masala Paneer Roast\', \'OliverÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x83Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82Â¢Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x80Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x99s Chettinad Aloo Roast\', \'OliverÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x83Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82Â¢Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x80Ã\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x99s Pudhina Aloo Roast\', \'Onion Pakoda\', \'Paneer Pakoda\', \'Egg Bhurji\', \'Masala Egg Pakoda\', \'Masala Omlette\', \'Egg Bhurji With Bacon & Sausage\', \'Chilly Egg\', \'Egg Bhurji With Bacon And Sausage\', \'Pizza Margharita\', \'Spicy Louisiana Pizza\', \'Wilted Spinach Pizza\', \'Eggplant Pizza\', \'Patata Salsa Pizza\', \'Stirfried Babycorn Pizza\', \'Arugula Pizza\', \'Pesto Shrooms Pizza\', \'Chilly Paneer Pizza\', \'Spicy Peri Peri Paneer & Spinach Pizza\', \'Mexican Nachos Special Pizza\', \'Chicken BBQ Pizza\', \'Chicken Sausage Pizza\', \'Chilly Chicken Pizza\', \'Chicken 65 Pizza\', \'Chicken Mexicana Pizza\', \'Chicken Bolognese Pizza\', \'Mexican Chicken Nachos Pizza\', \'Mexican Beef Nachos Pizza\', \'Arugula Bacon Pizza\', \'Pepperoni Pizza\', \'Chorizo Pizza\', \'Chocolate Brownie\']',
           "['Dal Tadka', 'Dal Makhani', 'Paneer Tikka Masala', 'Kadai Paneer', 'Paneer Makhani', 'Malai Kofta', 'Mutton Rogan Josh', 'Jeera Rice', 'Veg Biryani', 'Murgh Dum Biryani', 'Murgh Dum Biryani [Family Pack]', 'Hara Bara Kabab', 'Paneer Tikka', 'Tandoori Murgh', 'Murgh Lahori Kalmi', 'Jalebi with Rabdi', 'Roomali Roti', 'Afghani Naan', 'Stuffed Paratha', 'Paneer Tikka Masala Meal', 'Paneer Makhani Meal', 'Veg Biryani Meal', 'Executive Veg Box Meal', 'Executive Non Veg Box Meal', 'Rajma Meal', 'Peshawari Channa Meal', 'Paneer Tikka Masala Meal', 'Paneer Makhani Meal', 'Egg Curry Meal', 'Butter Chicken Meal', 'Rara Murgh Meal', 'Mutton Rogan Josh Meal', 'Rara Mutton Meal', 'Veg Biryani Meal', 'Egg Biryani Meal', 'Murgh Dum Biryani Meal', 'Murgh Tikka Biryani Meal', 'Mutton Biryani Meal', 'Prawns Biryani Meal', 'Executive Veg Box Meal', 'Executive Non Veg Box Meal', 'Tomato Shorba', 'Murgh Shorba', 'Bagicha ka Salad', 'Karela Salad', 'Kaju Fry Salad', 'Murgh Tikka Salad', 'French Fries', 'Aloo Tak-a-Tak', 'Baby Corn Harimirch Wala', 'Bhatti ka Gobi', 'Hara Bara Kabab', 'Tandoori Mushroom', 'Mushroom Harimirch Wala', 'Paneer Resunga', 'Paneer Tikka', 'Peshawari Seekh Kabab', 'Tandoori Murgh', 'Murgh Lahori Kalmi', 'Mutton Pepper Dry', 'Macchi Amritsari', 'Macchi Tak-a-Tak', 'Tawa Macchi', 'Prawns Balaika', 'Prawns Tak-a-Tak', 'Tandoori Prawns', 'Prawns Harimirch Wala', 'Dal Tadka', 'Dal Makhani', 'Paneer Tikka Masala', 'Paneer Saagwala', 'Kadai Paneer', 'Paneer Makhani', 'Mushroom Matar', 'Veg Patiala', 'Diwani Handi', 'Kadai Veg', 'Rajma', 'Sarson ka Saag', 'Aloo Gobi Masala', 'Bhindi Do Pyaza', 'Peshawari Channa', 'Malai Kofta', 'Kaju Masala', 'Egg Bhurji', 'Egg Masala', 'Rarra Mutton', 'Mutton Rogan Josh', 'Methi Macchi Masala', 'Macchi Jalfrezi', 'Prawns Masala', 'Plain Rice', 'Curd Rice', 'Jeera Rice', 'Veg Pulao', 'Peas Pulao', 'Veg Biryani', 'Egg Biryani', 'Murgh Dum Biryani', 'Murgh Tikka Biryani', 'Mutton Biryani', 'Prawns Biryani', 'Egg Biryani [Family Pack]', 'Murgh Dum Biryani [Family Pack]', 'Murgh Tikka Biryani [Family Pack]', 'Mutton Biryani [Family Pack]', 'Prawns Biryani [Family Pack]', 'Phulka', 'Roti', 'Butter Roti', 'Harimirch Paratha', 'Lalmirch Paratha', 'Garlic Naan', 'Naan', 'Butter Naan', 'Lachha Paratha', 'Kulcha', 'Butter Kulcha', 'Makkai Ki Roti', 'Roomali Roti', 'Afghani Naan', 'Peshawari Paratha', 'Stuffed Paratha', '8 Roti ka Chota Khazana', '12 Roti ka Bada Khazana', 'Egg Paratha', 'Mutton Keema Paratha', 'Roasted Papad', 'Masala Papad', 'Plain Curd', 'Mixed Veg Raita', 'Boondi Raita', 'Cucumber Raita', 'Pineapple Raita', 'Onion Raita', 'Mint Raita', 'Rabdi', 'Jalebi', 'Jalebi with Rabdi', 'Gulab Jamun', 'Rasgulla', 'Bhune Jeere ki Chaach', 'Jal Jeera', 'Meetha Punjabi Lassi', 'Namkeen Punjabi Lassi', 'Masala Punjabi Lassi', 'Patiala Punjabi Lassi', 'Meetha Nimboo Paani', 'Namkeen Nimboo Paani', 'Roohafza Sherbat', 'Mineral Water [1 litre]']"],
          dtype=object)




```python
df["rest_type"].unique()
```




    array(['Casual Dining', 'Cafe, Casual Dining', 'Quick Bites',
           'Casual Dining, Cafe', 'Cafe', 'Quick Bites, Cafe',
           'Cafe, Quick Bites', 'Delivery', 'Mess', 'Dessert Parlor',
           'Bakery, Dessert Parlor', 'Pub', 'Bakery', 'Takeaway, Delivery',
           'Fine Dining', 'Beverage Shop', 'Sweet Shop', 'Bar',
           'Dessert Parlor, Sweet Shop', 'Bakery, Quick Bites',
           'Sweet Shop, Quick Bites', 'Kiosk', 'Food Truck',
           'Quick Bites, Dessert Parlor', 'Beverage Shop, Quick Bites',
           'Beverage Shop, Dessert Parlor', 'Takeaway', 'Pub, Casual Dining',
           'Casual Dining, Bar', 'Dessert Parlor, Beverage Shop',
           'Quick Bites, Bakery', 'Microbrewery, Casual Dining', 'Lounge',
           'Bar, Casual Dining', 'Food Court', 'Cafe, Bakery', nan, 'Dhaba',
           'Quick Bites, Sweet Shop', 'Microbrewery',
           'Food Court, Quick Bites', 'Quick Bites, Beverage Shop',
           'Pub, Bar', 'Casual Dining, Pub', 'Lounge, Bar',
           'Dessert Parlor, Quick Bites', 'Food Court, Dessert Parlor',
           'Casual Dining, Sweet Shop', 'Food Court, Casual Dining',
           'Casual Dining, Microbrewery', 'Lounge, Casual Dining',
           'Cafe, Food Court', 'Beverage Shop, Cafe', 'Cafe, Dessert Parlor',
           'Dessert Parlor, Cafe', 'Dessert Parlor, Bakery',
           'Microbrewery, Pub', 'Bakery, Food Court', 'Club',
           'Quick Bites, Food Court', 'Bakery, Cafe', 'Pub, Cafe',
           'Casual Dining, Irani Cafee', 'Fine Dining, Lounge',
           'Bar, Quick Bites', 'Confectionery', 'Pub, Microbrewery',
           'Microbrewery, Lounge', 'Fine Dining, Microbrewery',
           'Fine Dining, Bar', 'Dessert Parlor, Kiosk', 'Bhojanalya',
           'Casual Dining, Quick Bites', 'Cafe, Bar', 'Casual Dining, Lounge',
           'Bakery, Beverage Shop', 'Microbrewery, Bar', 'Cafe, Lounge',
           'Bar, Pub', 'Lounge, Cafe', 'Club, Casual Dining',
           'Quick Bites, Mess', 'Quick Bites, Meat Shop',
           'Quick Bites, Kiosk', 'Lounge, Microbrewery',
           'Food Court, Beverage Shop', 'Dessert Parlor, Food Court',
           'Bar, Lounge'], dtype=object)




```python
df["rest_type"].nunique()
```




    87




```python
df["rate"].nunique()
```




    62




```python
rate =df["rate"].unique()
```


```python
rate
```




    array(['4.1/5', '3.8/5', '3.7/5', '3.6/5', '4.6/5', '4.0/5', '4.2/5',
           '3.9/5', '3.1/5', '3.0/5', '3.2/5', '3.3/5', '2.8/5', '4.4/5',
           '4.3/5', '2.9/5', '3.5/5', '2.6/5', '3.8 /5', '3.4/5', '4.5/5',
           '2.5/5', '2.7/5', '4.7/5', '2.4/5', '2.2/5', '2.3/5', '3.4 /5',
           '3.6 /5', '4.8/5', '3.9 /5', '4.2 /5', '4.0 /5', '4.1 /5',
           '3.7 /5', '3.1 /5', '2.9 /5', '3.3 /5', '2.8 /5', '3.5 /5',
           '2.7 /5', '2.5 /5', '3.2 /5', '2.6 /5', '4.5 /5', '4.3 /5',
           '4.4 /5', '4.9/5', '2.1/5', '2.0/5', '1.8/5', '4.6 /5', '4.9 /5',
           '3.0 /5', '4.8 /5', '2.3 /5', '4.7 /5', '2.4 /5', '2.1 /5',
           '2.2 /5', '2.0 /5', '1.8 /5'], dtype=object)




```python
df[df["rate"]=='-'].shape
```




    (0, 17)




```python
df['rating'] = df['rate'].str.extract(r'(\d+\.\d+)')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
```


```python
df.dtypes
```




    url                             object
    address                         object
    name                            object
    online_order                    object
    book_table                      object
    rate                            object
    votes                            int64
    phone                           object
    location                        object
    rest_type                       object
    dish_liked                      object
    cuisines                        object
    approx_cost(for two people)     object
    reviews_list                    object
    menu_item                       object
    listed_in(type)                 object
    listed_in(city)                 object
    rating                         float64
    dtype: object



### Removing the Cost column data that contains None and lest than 100


```python
df =df[df["approx_cost(for two people)"].notna() ]
```


```python
df["approx_cost(for two people)"].unique()
```




    array(['800', '300', '600', '700', '550', '500', '450', '650', '400',
           '900', '200', '750', '150', '850', '100', '1,200', '350', '250',
           '950', '1,000', '1,500', '1,300', '199', '1,100', '1,600', '230',
           '130', '1,700', '1,350', '2,200', '1,400', '2,000', '1,800',
           '1,900', '180', '330', '2,500', '2,100', '3,000', '2,800', '3,400',
           '50', '40', '1,250', '3,500', '4,000', '2,400', '2,600', '1,450',
           '70', '3,200', '240', '6,000', '1,050', '2,300', '4,100', '120',
           '5,000', '3,700', '1,650', '2,700', '4,500', '80'], dtype=object)



### Converting to numeric by removing comma from the cost


```python
df['Cost_For_two'] = df['approx_cost(for two people)'].replace(',', '', regex=True)  # Remove commas
df['Cost_For_two'] = pd.to_numeric(df['Cost_For_two'], errors='coerce')  # Convert to numeric

```


```python
(df[df['Cost_For_two'] <200].shape)[0]
```




    1998




```python
df = df[df['Cost_For_two'] >100]
```


```python
df["approx_cost(for two people)"].nunique()
```




    58




```python
cost =df["Cost_For_two"].unique()
```


```python
cost
```




    array([ 800,  300,  600,  700,  550,  500,  450,  650,  400,  900,  200,
            750,  150,  850, 1200,  350,  250,  950, 1000, 1500, 1300,  199,
           1100, 1600,  230,  130, 1700, 1350, 2200, 1400, 2000, 1800, 1900,
            180,  330, 2500, 2100, 3000, 2800, 3400, 1250, 3500, 4000, 2400,
           2600, 1450, 3200,  240, 6000, 1050, 2300, 4100,  120, 5000, 3700,
           1650, 2700, 4500])




```python
df.columns
```




    Index(['url', 'address', 'name', 'online_order', 'book_table', 'rate', 'votes',
           'phone', 'location', 'rest_type', 'dish_liked', 'cuisines',
           'approx_cost(for two people)', 'reviews_list', 'menu_item',
           'listed_in(type)', 'listed_in(city)', 'rating', 'Cost_For_two'],
          dtype='object')




```python
# Exclude specific columns
df = df.drop(columns=['approx_cost(for two people)', 'rate'])

```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st Main Road, 2nd Stage, Banashankari, ...</td>
      <td>Jalsa</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>775</td>
      <td>080 42297555\r\n+91 9743772233</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Pasta, Lunch Buffet, Masala Papad, Paneer Laja...</td>
      <td>North Indian, Mughlai, Chinese</td>
      <td>[('Rated 4.0', 'RATED\n  A beautiful place to ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...</td>
      <td>Spice Elephant</td>
      <td>Yes</td>
      <td>No</td>
      <td>787</td>
      <td>080 41714161</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Momos, Lunch Buffet, Chocolate Nirvana, Thai G...</td>
      <td>Chinese, North Indian, Thai</td>
      <td>[('Rated 4.0', 'RATED\n  Had been here for din...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/SanchurroBangalore?cont...</td>
      <td>1112, Next to KIMS Medical College, 17th Cross...</td>
      <td>San Churro Cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>918</td>
      <td>+91 9663487993</td>
      <td>Banashankari</td>
      <td>Cafe, Casual Dining</td>
      <td>Churros, Cannelloni, Minestrone Soup, Hot Choc...</td>
      <td>Cafe, Mexican, Italian</td>
      <td>[('Rated 3.0', "RATED\n  Ambience is not that ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st Floor, Annakuteera, 3rd Stage, Banashankar...</td>
      <td>Addhuri Udupi Bhojana</td>
      <td>No</td>
      <td>No</td>
      <td>88</td>
      <td>+91 9620009302</td>
      <td>Banashankari</td>
      <td>Quick Bites</td>
      <td>Masala Dosa</td>
      <td>South Indian, North Indian</td>
      <td>[('Rated 4.0', "RATED\n  Great food and proper...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.7</td>
      <td>300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd Floor, Lakshmi Associates, Gandhi Baza...</td>
      <td>Grand Village</td>
      <td>No</td>
      <td>No</td>
      <td>166</td>
      <td>+91 8026612447\r\n+91 9901210005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Panipuri, Gol Gappe</td>
      <td>North Indian, Rajasthani</td>
      <td>[('Rated 4.0', 'RATED\n  Very good restaurant ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>600</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51709</th>
      <td>https://www.zomato.com/bangalore/the-farm-hous...</td>
      <td>136, SAP Labs India, KIADB Export Promotion In...</td>
      <td>The Farm House Bar n Grill</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>+91 9980121279\n+91 9900240646</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>NaN</td>
      <td>North Indian, Continental</td>
      <td>[('Rated 4.0', 'RATED\n  Ambience- Big and spa...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.7</td>
      <td>800</td>
    </tr>
    <tr>
      <th>51711</th>
      <td>https://www.zomato.com/bangalore/bhagini-2-whi...</td>
      <td>139/C1, Next To GR Tech Park, Pattandur Agraha...</td>
      <td>Bhagini</td>
      <td>No</td>
      <td>No</td>
      <td>81</td>
      <td>080 65951222</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>Biryani, Andhra Meal</td>
      <td>Andhra, South Indian, Chinese, North Indian</td>
      <td>[('Rated 4.0', 'RATED\n  A fine place to chill...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>2.5</td>
      <td>800</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>Four Points by Sheraton Bengaluru, 43/3, White...</td>
      <td>Best Brews - Four Points by Sheraton Bengaluru...</td>
      <td>No</td>
      <td>No</td>
      <td>27</td>
      <td>080 40301477</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>Continental</td>
      <td>[('Rated 5.0', "RATED\n  Food and service are ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.6</td>
      <td>1500</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>Chime - Sheraton Grand Bengaluru Whitefield Ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>Finger Food</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>4.3</td>
      <td>2500</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>ITPL Main Road, KIADB Export Promotion Industr...</td>
      <td>The Nest - The Den Bengaluru</td>
      <td>No</td>
      <td>No</td>
      <td>13</td>
      <td>+91 8071117272</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar, Casual Dining</td>
      <td>NaN</td>
      <td>Finger Food, North Indian, Continental</td>
      <td>[('Rated 5.0', 'RATED\n  Great ambience , look...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.4</td>
      <td>1500</td>
    </tr>
  </tbody>
</table>
<p>40764 rows × 17 columns</p>
</div>



## Normalizing the Name , cuisine columns and creating Cuisine_Count



```python

# Specify the columns you want to normalize
columns_to_normalize = ['name', 'cuisines']

# Convert specified columns to lowercase
for col in columns_to_normalize:
    df[col] = df[col].str.lower()



```


```python
# Handle None values by applying a condition
df['Cuisine_Count'] = df['cuisines'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st Main Road, 2nd Stage, Banashankari, ...</td>
      <td>jalsa</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>775</td>
      <td>080 42297555\r\n+91 9743772233</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Pasta, Lunch Buffet, Masala Papad, Paneer Laja...</td>
      <td>north indian, mughlai, chinese</td>
      <td>[('Rated 4.0', 'RATED\n  A beautiful place to ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...</td>
      <td>spice elephant</td>
      <td>Yes</td>
      <td>No</td>
      <td>787</td>
      <td>080 41714161</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Momos, Lunch Buffet, Chocolate Nirvana, Thai G...</td>
      <td>chinese, north indian, thai</td>
      <td>[('Rated 4.0', 'RATED\n  Had been here for din...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/SanchurroBangalore?cont...</td>
      <td>1112, Next to KIMS Medical College, 17th Cross...</td>
      <td>san churro cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>918</td>
      <td>+91 9663487993</td>
      <td>Banashankari</td>
      <td>Cafe, Casual Dining</td>
      <td>Churros, Cannelloni, Minestrone Soup, Hot Choc...</td>
      <td>cafe, mexican, italian</td>
      <td>[('Rated 3.0', "RATED\n  Ambience is not that ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st Floor, Annakuteera, 3rd Stage, Banashankar...</td>
      <td>addhuri udupi bhojana</td>
      <td>No</td>
      <td>No</td>
      <td>88</td>
      <td>+91 9620009302</td>
      <td>Banashankari</td>
      <td>Quick Bites</td>
      <td>Masala Dosa</td>
      <td>south indian, north indian</td>
      <td>[('Rated 4.0', "RATED\n  Great food and proper...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.7</td>
      <td>300</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd Floor, Lakshmi Associates, Gandhi Baza...</td>
      <td>grand village</td>
      <td>No</td>
      <td>No</td>
      <td>166</td>
      <td>+91 8026612447\r\n+91 9901210005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Panipuri, Gol Gappe</td>
      <td>north indian, rajasthani</td>
      <td>[('Rated 4.0', 'RATED\n  Very good restaurant ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>600</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51709</th>
      <td>https://www.zomato.com/bangalore/the-farm-hous...</td>
      <td>136, SAP Labs India, KIADB Export Promotion In...</td>
      <td>the farm house bar n grill</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>+91 9980121279\n+91 9900240646</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>NaN</td>
      <td>north indian, continental</td>
      <td>[('Rated 4.0', 'RATED\n  Ambience- Big and spa...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.7</td>
      <td>800</td>
      <td>2</td>
    </tr>
    <tr>
      <th>51711</th>
      <td>https://www.zomato.com/bangalore/bhagini-2-whi...</td>
      <td>139/C1, Next To GR Tech Park, Pattandur Agraha...</td>
      <td>bhagini</td>
      <td>No</td>
      <td>No</td>
      <td>81</td>
      <td>080 65951222</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>Biryani, Andhra Meal</td>
      <td>andhra, south indian, chinese, north indian</td>
      <td>[('Rated 4.0', 'RATED\n  A fine place to chill...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>2.5</td>
      <td>800</td>
      <td>4</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>Four Points by Sheraton Bengaluru, 43/3, White...</td>
      <td>best brews - four points by sheraton bengaluru...</td>
      <td>No</td>
      <td>No</td>
      <td>27</td>
      <td>080 40301477</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>continental</td>
      <td>[('Rated 5.0', "RATED\n  Food and service are ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.6</td>
      <td>1500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>chime - sheraton grand bengaluru whitefield ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>finger food</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>4.3</td>
      <td>2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>ITPL Main Road, KIADB Export Promotion Industr...</td>
      <td>the nest - the den bengaluru</td>
      <td>No</td>
      <td>No</td>
      <td>13</td>
      <td>+91 8071117272</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar, Casual Dining</td>
      <td>NaN</td>
      <td>finger food, north indian, continental</td>
      <td>[('Rated 5.0', 'RATED\n  Great ambience , look...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.4</td>
      <td>1500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>40764 rows × 18 columns</p>
</div>




```python
# Find the row with the maximum number of cuisines
max_cuisine_row = df.loc[df['Cuisine_Count'].idxmax()]
max_cuisine_row
```




    url                https://www.zomato.com/bangalore/freshmenu-ban...
    address            10, Stavyah Arcade, 3rd Floor 9th Main, Yarab ...
    name                                                       freshmenu
    online_order                                                     Yes
    book_table                                                        No
    votes                                                            627
    phone                                                   080 40424242
    location                                                Banashankari
    rest_type                                                   Delivery
    dish_liked         Salads, Sandwiches, Salad, Thai Rice, Pasta, N...
    cuisines           healthy food, chinese, biryani, north indian, ...
    reviews_list       [('Rated 5.0', 'RATED\n  What: continental foo...
    menu_item          ["Egg 'n' Chicken Ham Breakwich", 'Maple Panca...
    listed_in(type)                                             Delivery
    listed_in(city)                                         Banashankari
    rating                                                           3.9
    Cost_For_two                                                     450
    Cuisine_Count                                                      8
    Name: 55, dtype: object




```python
# Find the row with the maximum number of cuisines
min_cuisine_row = df.loc[df['Cuisine_Count'].idxmin()]
min_cuisine_row
```




    url                https://www.zomato.com/bangalore/noodle-oodle-...
    address            V3/1, NGEF, Industrial Estate, Mahadevapura Po...
    name                                                    noodle oodle
    online_order                                                     Yes
    book_table                                                        No
    votes                                                              9
    phone                                                 +91 9945670505
    location                                                  Whitefield
    rest_type                                                   Delivery
    dish_liked                                                       NaN
    cuisines                                                         NaN
    reviews_list       [('Rated 4.0', 'RATED\n  Lil oily else good'),...
    menu_item          ['Chilly Chicken', 'Mixed Veg Noodles', 'Veg S...
    listed_in(type)                                             Delivery
    listed_in(city)                                          Brookefield
    rating                                                           3.6
    Cost_For_two                                                     400
    Cuisine_Count                                                      0
    Name: 6887, dtype: object




```python
# Six hotels with 0 cuisines
(df[df['Cuisine_Count'] ==0].shape)[0]
```




    6




```python
df[df['Cuisine_Count'] ==0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6887</th>
      <td>https://www.zomato.com/bangalore/noodle-oodle-...</td>
      <td>V3/1, NGEF, Industrial Estate, Mahadevapura Po...</td>
      <td>noodle oodle</td>
      <td>Yes</td>
      <td>No</td>
      <td>9</td>
      <td>+91 9945670505</td>
      <td>Whitefield</td>
      <td>Delivery</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[('Rated 4.0', 'RATED\n  Lil oily else good'),...</td>
      <td>['Chilly Chicken', 'Mixed Veg Noodles', 'Veg S...</td>
      <td>Delivery</td>
      <td>Brookefield</td>
      <td>3.6</td>
      <td>400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24725</th>
      <td>https://www.zomato.com/bangalore/swagatham-ray...</td>
      <td>604, 2nd Block, HBR Layout Kalyan Nagar, Banga...</td>
      <td>swagatham rayalaseema ruchulu</td>
      <td>Yes</td>
      <td>No</td>
      <td>24</td>
      <td>+91 9986222755</td>
      <td>Kalyan Nagar</td>
      <td>Casual Dining</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[('Rated 1.0', "RATED\n  Mediocre!\n\nWe went ...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Kalyan Nagar</td>
      <td>3.3</td>
      <td>600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26186</th>
      <td>https://www.zomato.com/bangalore/swagatham-ray...</td>
      <td>604, 2nd Block, HBR Layout Kalyan Nagar, Banga...</td>
      <td>swagatham rayalaseema ruchulu</td>
      <td>Yes</td>
      <td>No</td>
      <td>24</td>
      <td>+91 9986222755</td>
      <td>Kalyan Nagar</td>
      <td>Casual Dining</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[('Rated 1.0', "RATED\n  Mediocre!\n\nWe went ...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Kammanahalli</td>
      <td>3.3</td>
      <td>600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40625</th>
      <td>https://www.zomato.com/bangalore/noodle-oodle-...</td>
      <td>V3/1, NGEF, Industrial Estate, Mahadevapura Po...</td>
      <td>noodle oodle</td>
      <td>Yes</td>
      <td>No</td>
      <td>12</td>
      <td>080 49652930</td>
      <td>Whitefield</td>
      <td>Delivery</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[('Rated 4.0', 'RATED\n  Lil oily else good'),...</td>
      <td>['Gobi Manchurian', 'Chilly Chicken', 'Mixed V...</td>
      <td>Delivery</td>
      <td>Marathahalli</td>
      <td>3.7</td>
      <td>400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50355</th>
      <td>https://www.zomato.com/bangalore/noodle-oodle-...</td>
      <td>V3/1, NGEF, Industrial Estate, Mahadevapura Po...</td>
      <td>noodle oodle</td>
      <td>Yes</td>
      <td>No</td>
      <td>12</td>
      <td>080 49652930</td>
      <td>Whitefield</td>
      <td>Delivery</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[('Rated 4.0', 'RATED\n  Lil oily else good'),...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Whitefield</td>
      <td>3.7</td>
      <td>400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50439</th>
      <td>https://www.zomato.com/bangalore/taste-of-chet...</td>
      <td>V3/1, NGEF, Industrial Estate, Mahadevapura Po...</td>
      <td>taste of chettinad</td>
      <td>Yes</td>
      <td>No</td>
      <td>6</td>
      <td>+91 9595783578</td>
      <td>Whitefield</td>
      <td>Delivery</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>[('Rated 1.0', 'RATED\n  The biryani was not e...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Whitefield</td>
      <td>3.2</td>
      <td>400</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Six hotels with 0 cuisines
(df[df['Cuisine_Count'] ==1].shape)[0]
```




    8455




```python
df[df['Cuisine_Count'] ==1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>https://www.zomato.com/bangalore/timepass-dinn...</td>
      <td>37, 5-1, 4th Floor, Bosco Court, Gandhi Bazaar...</td>
      <td>timepass dinner</td>
      <td>Yes</td>
      <td>No</td>
      <td>286</td>
      <td>+91 9980040002\r\n+91 9980063005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Onion Rings, Pasta, Kadhai Paneer, Salads, Sal...</td>
      <td>north indian</td>
      <td>[('Rated 3.0', 'RATED\n  Food 3/5\nAmbience 3/...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>600</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>https://www.zomato.com/bangalore/caf%C3%A9-dow...</td>
      <td>12,29 Near PES University Back Gate, D'Souza N...</td>
      <td>cafãâãâãâãâãâãâãâãâ© down the a...</td>
      <td>Yes</td>
      <td>No</td>
      <td>402</td>
      <td>080 26724489\r\n+91 7406048982</td>
      <td>Banashankari</td>
      <td>Cafe</td>
      <td>Waffles, Pasta, Crispy Chicken, Honey Chilli C...</td>
      <td>cafe</td>
      <td>[('Rated 4.0', 'RATED\n  We ended up here on a...</td>
      <td>[]</td>
      <td>Cafes</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>https://www.zomato.com/bangalore/cafe-vivacity...</td>
      <td>2303, 21st Cross, K R Road, 2nd Stage, Banasha...</td>
      <td>cafe vivacity</td>
      <td>Yes</td>
      <td>No</td>
      <td>90</td>
      <td>080 26768182\r\n+91 9845704455</td>
      <td>Banashankari</td>
      <td>Cafe</td>
      <td>Garlic Bread, Burgers, Sandwiches, Pizza, Hot ...</td>
      <td>cafe</td>
      <td>[('Rated 2.0', 'RATED\n  Not so good place as ...</td>
      <td>[]</td>
      <td>Cafes</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>650</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>https://www.zomato.com/bangalore/hide-out-cafe...</td>
      <td>775/1, Opposite Gupta Collage, 7th Block, 3rd ...</td>
      <td>hide out cafe</td>
      <td>No</td>
      <td>No</td>
      <td>31</td>
      <td>+91 9901481185</td>
      <td>Banashankari</td>
      <td>Cafe</td>
      <td>NaN</td>
      <td>cafe</td>
      <td>[('Rated 4.0', 'RATED\n  The food was quite go...</td>
      <td>[]</td>
      <td>Cafes</td>
      <td>Banashankari</td>
      <td>3.7</td>
      <td>300</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>https://www.zomato.com/bangalore/ovenstory-piz...</td>
      <td>101, Ground Floor, Manjunatha Complex, 22nd Ma...</td>
      <td>ovenstory pizza</td>
      <td>Yes</td>
      <td>No</td>
      <td>172</td>
      <td>+91 7738383000</td>
      <td>Banashankari</td>
      <td>Delivery</td>
      <td>Paneer Tikka, Garlic Bread, Thin Crust Pizza, ...</td>
      <td>pizza</td>
      <td>[('Rated 4.0', 'RATED\n  Stumbled upon this on...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Banashankari</td>
      <td>3.9</td>
      <td>750</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51681</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>chime - sheraton grand bengaluru whitefield ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>finger food</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Drinks &amp; nightlife</td>
      <td>Whitefield</td>
      <td>4.3</td>
      <td>2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51683</th>
      <td>https://www.zomato.com/bangalore/alt-whitefiel...</td>
      <td>Sky Deck, VR Bengaluru, Whitefield Main Road, ...</td>
      <td>alt</td>
      <td>No</td>
      <td>Yes</td>
      <td>821</td>
      <td>080 49653221</td>
      <td>Whitefield</td>
      <td>Bar, Lounge</td>
      <td>Margarita, Lamb, Cocktails, Nachos, Spring Rol...</td>
      <td>finger food</td>
      <td>[('Rated 3.0', "RATED\n  It's a skydeck which ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>4.1</td>
      <td>1900</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51707</th>
      <td>https://www.zomato.com/bangalore/m-bar-bengalu...</td>
      <td>Bengaluru Marriott Hotel, 75, 8th Road, EPIP A...</td>
      <td>m bar - bengaluru marriott hotel whitefield</td>
      <td>No</td>
      <td>No</td>
      <td>77</td>
      <td>080 49435000</td>
      <td>Whitefield</td>
      <td>Fine Dining, Bar</td>
      <td>Rooftop Ambience</td>
      <td>finger food</td>
      <td>[('Rated 4.0', 'RATED\n  Went there post dinne...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.9</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>Four Points by Sheraton Bengaluru, 43/3, White...</td>
      <td>best brews - four points by sheraton bengaluru...</td>
      <td>No</td>
      <td>No</td>
      <td>27</td>
      <td>080 40301477</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>continental</td>
      <td>[('Rated 5.0', "RATED\n  Food and service are ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.6</td>
      <td>1500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>chime - sheraton grand bengaluru whitefield ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>finger food</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>4.3</td>
      <td>2500</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>8455 rows × 18 columns</p>
</div>




```python
(df[df['menu_item']=="[]"].shape)[0]
```




    29947




```python
df['reviews_list'][0]
```




    '[(\'Rated 4.0\', \'RATED\\n  A beautiful place to dine in.The interiors take you back to the Mughal era. The lightings are just perfect.We went there on the occasion of Christmas and so they had only limited items available. But the taste and service was not compromised at all.The only complaint is that the breads could have been better.Would surely like to come here again.\'), (\'Rated 4.0\', \'RATED\\n  I was here for dinner with my family on a weekday. The restaurant was completely empty. Ambience is good with some good old hindi music. Seating arrangement are good too. We ordered masala papad, panner and baby corn starters, lemon and corrionder soup, butter roti, olive and chilli paratha. Food was fresh and good, service is good too. Good for family hangout.\\nCheers\'), (\'Rated 2.0\', \'RATED\\n  Its a restaurant near to Banashankari BDA. Me along with few of my office friends visited to have buffet but unfortunately they only provide veg buffet. On inquiring they said this place is mostly visited by vegetarians. Anyways we ordered ala carte items which took ages to come. Food was ok ok. Definitely not visiting anymore.\'), (\'Rated 4.0\', \'RATED\\n  We went here on a weekend and one of us had the buffet while two of us took Ala Carte. Firstly the ambience and service of this place is great! The buffet had a lot of items and the good was good. We had a Pumpkin Halwa intm the dessert which was amazing. Must try! The kulchas are great here. Cheers!\'), (\'Rated 5.0\', \'RATED\\n  The best thing about the place is itÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x92s ambiance. Second best thing was yummy ? food. We try buffet and buffet food was not disappointed us.\\nTest ?. ?? ?? ?? ?? ??\\nQuality ?. ??????????.\\nService: Staff was very professional and friendly.\\n\\nOverall experience was excellent.\\n\\nsubirmajumder85.wixsite.com\'), (\'Rated 5.0\', \'RATED\\n  Great food and pleasant ambience. Expensive but Coll place to chill and relax......\\n\\nService is really very very good and friendly staff...\\n\\nFood : 5/5\\nService : 5/5\\nAmbience :5/5\\nOverall :5/5\'), (\'Rated 4.0\', \'RATED\\n  Good ambience with tasty food.\\nCheese chilli paratha with Bhutta palak methi curry is a good combo.\\nLemon Chicken in the starters is a must try item.\\nEgg fried rice was also quite tasty.\\nIn the mocktails, recommend "Alice in Junoon". Do not miss it.\'), (\'Rated 4.0\', \'RATED\\n  You canÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x92t go wrong with Jalsa. Never been a fan of their buffet and thus always order alacarteÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x92. Service at times can be on the slower side but food is worth the wait.\'), (\'Rated 5.0\', \'RATED\\n  Overdelighted by the service and food provided at this place. A royal and ethnic atmosphere builds a strong essence of being in India and also the quality and taste of food is truly authentic. I would totally recommend to visit this place once.\'), (\'Rated 4.0\', \'RATED\\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\\n\\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome\'), (\'Rated 4.0\', \'RATED\\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\\n\\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome\'), (\'Rated 4.0\', \'RATED\\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\\n\\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome\')]'




```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st Main Road, 2nd Stage, Banashankari, ...</td>
      <td>jalsa</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>775</td>
      <td>080 42297555\r\n+91 9743772233</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Pasta, Lunch Buffet, Masala Papad, Paneer Laja...</td>
      <td>north indian, mughlai, chinese</td>
      <td>[('Rated 4.0', 'RATED\n  A beautiful place to ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...</td>
      <td>spice elephant</td>
      <td>Yes</td>
      <td>No</td>
      <td>787</td>
      <td>080 41714161</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Momos, Lunch Buffet, Chocolate Nirvana, Thai G...</td>
      <td>chinese, north indian, thai</td>
      <td>[('Rated 4.0', 'RATED\n  Had been here for din...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/SanchurroBangalore?cont...</td>
      <td>1112, Next to KIMS Medical College, 17th Cross...</td>
      <td>san churro cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>918</td>
      <td>+91 9663487993</td>
      <td>Banashankari</td>
      <td>Cafe, Casual Dining</td>
      <td>Churros, Cannelloni, Minestrone Soup, Hot Choc...</td>
      <td>cafe, mexican, italian</td>
      <td>[('Rated 3.0', "RATED\n  Ambience is not that ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st Floor, Annakuteera, 3rd Stage, Banashankar...</td>
      <td>addhuri udupi bhojana</td>
      <td>No</td>
      <td>No</td>
      <td>88</td>
      <td>+91 9620009302</td>
      <td>Banashankari</td>
      <td>Quick Bites</td>
      <td>Masala Dosa</td>
      <td>south indian, north indian</td>
      <td>[('Rated 4.0', "RATED\n  Great food and proper...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.7</td>
      <td>300</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd Floor, Lakshmi Associates, Gandhi Baza...</td>
      <td>grand village</td>
      <td>No</td>
      <td>No</td>
      <td>166</td>
      <td>+91 8026612447\r\n+91 9901210005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Panipuri, Gol Gappe</td>
      <td>north indian, rajasthani</td>
      <td>[('Rated 4.0', 'RATED\n  Very good restaurant ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>600</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51709</th>
      <td>https://www.zomato.com/bangalore/the-farm-hous...</td>
      <td>136, SAP Labs India, KIADB Export Promotion In...</td>
      <td>the farm house bar n grill</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>+91 9980121279\n+91 9900240646</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>NaN</td>
      <td>north indian, continental</td>
      <td>[('Rated 4.0', 'RATED\n  Ambience- Big and spa...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.7</td>
      <td>800</td>
      <td>2</td>
    </tr>
    <tr>
      <th>51711</th>
      <td>https://www.zomato.com/bangalore/bhagini-2-whi...</td>
      <td>139/C1, Next To GR Tech Park, Pattandur Agraha...</td>
      <td>bhagini</td>
      <td>No</td>
      <td>No</td>
      <td>81</td>
      <td>080 65951222</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>Biryani, Andhra Meal</td>
      <td>andhra, south indian, chinese, north indian</td>
      <td>[('Rated 4.0', 'RATED\n  A fine place to chill...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>2.5</td>
      <td>800</td>
      <td>4</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>Four Points by Sheraton Bengaluru, 43/3, White...</td>
      <td>best brews - four points by sheraton bengaluru...</td>
      <td>No</td>
      <td>No</td>
      <td>27</td>
      <td>080 40301477</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>continental</td>
      <td>[('Rated 5.0', "RATED\n  Food and service are ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.6</td>
      <td>1500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>chime - sheraton grand bengaluru whitefield ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>finger food</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>4.3</td>
      <td>2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>ITPL Main Road, KIADB Export Promotion Industr...</td>
      <td>the nest - the den bengaluru</td>
      <td>No</td>
      <td>No</td>
      <td>13</td>
      <td>+91 8071117272</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar, Casual Dining</td>
      <td>NaN</td>
      <td>finger food, north indian, continental</td>
      <td>[('Rated 5.0', 'RATED\n  Great ambience , look...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.4</td>
      <td>1500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>40764 rows × 18 columns</p>
</div>




```python
df['listed_in(city)'].unique()
```




    array(['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
           'Brigade Road', 'Brookefield', 'BTM', 'Church Street',
           'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
           'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli',
           'Koramangala 4th Block', 'Koramangala 5th Block',
           'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road',
           'Malleshwaram', 'Marathahalli', 'MG Road', 'New BEL Road',
           'Old Airport Road', 'Rajajinagar', 'Residency Road',
           'Sarjapur Road', 'Whitefield'], dtype=object)




```python
df['listed_in(type)'].unique()
```




    array(['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
           'Drinks & nightlife', 'Pubs and bars'], dtype=object)




```python
!pip install seaborn
```

    Requirement already satisfied: seaborn in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (0.13.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from seaborn) (2.0.2)
    Requirement already satisfied: pandas>=1.2 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from seaborn) (2.2.3)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from seaborn) (3.9.3)
    Requirement already satisfied: contourpy>=1.0.1 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.55.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.7)
    Requirement already satisfied: packaging>=20.0 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.1)
    Requirement already satisfied: pillow>=8 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (11.0.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.0)
    Requirement already satisfied: python-dateutil>=2.7 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
    Requirement already satisfied: importlib-resources>=3.2.0 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (6.4.5)
    Requirement already satisfied: pytz>=2020.1 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from pandas>=1.2->seaborn) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from pandas>=1.2->seaborn) (2024.2)
    Requirement already satisfied: zipp>=3.1.0 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from importlib-resources>=3.2.0->matplotlib!=3.6.1,>=3.4->seaborn) (3.21.0)
    Requirement already satisfied: six>=1.5 in /home/srinath2003/miniconda3/envs/mongo/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.16.0)



```python
import seaborn as sns
import matplotlib.pyplot as plt

```


```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='listed_in(type)')
```




    <Axes: xlabel='listed_in(type)', ylabel='count'>




    
![png](output_60_1.png)
    



```python
plt.figure(figsize=(10, 6))
sns.histplot(data=df, hue='listed_in(type)', x='listed_in(city)', multiple='stack')
plt.xticks(rotation=90)
plt.show()
```


    
![png](output_61_0.png)
    



```python
sns.set_style("whitegrid")
palette = sns.color_palette("tab20", n_colors=df['listed_in(type)'].nunique())

freq_data = df.groupby(['listed_in(city)', 'listed_in(type)']).size().reset_index(name='count')
freq_data = freq_data.sort_values(['listed_in(city)', 'count'], ascending=[True, True])
plt.figure(figsize=(14, 8))
sns.barplot(
    data=freq_data,
    x='listed_in(city)',
    y='count',
    hue='listed_in(type)',
    dodge=True,
    palette=palette,
    edgecolor="black",
    alpha=0.8
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.xlabel('City', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Listings by City and Type (Adjacent Bars)', fontsize=16, weight='bold')

plt.legend(title='Listing Type', title_fontsize=13, fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
```


    
![png](output_62_0.png)
    



```python
# Filter the dataset to include only rows where 'listed_in(type)' is 'Drinks & nightlife'
filtered_df = df[df['listed_in(type)'] == 'Drinks & nightlife']
# Count occurrences and reorder the DataFrame by city counts
city_counts = filtered_df['listed_in(city)'].value_counts()
sorted_cities = city_counts.index
# Set 'listed_in(city)' as a categorical type with sorted order
filtered_df['listed_in(city)'] = pd.Categorical(filtered_df['listed_in(city)'], categories=sorted_cities, ordered=True)
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df, x='listed_in(city)', multiple='stack')
plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees for better readability
plt.title("Distribution of 'Drinks & Nightlife' by City (Sorted)")
plt.show()
```


    
![png](output_63_0.png)
    



```python
# Filter the dataset to include only rows where 'listed_in(type)' is 'Drinks & nightlife'
filtered_df = df[df['listed_in(type)'] == 'Pubs and bars']
# Count occurrences and reorder the DataFrame by city counts
city_counts = filtered_df['listed_in(city)'].value_counts()
sorted_cities = city_counts.index
# Set 'listed_in(city)' as a categorical type with sorted order
filtered_df['listed_in(city)'] = pd.Categorical(filtered_df['listed_in(city)'], categories=sorted_cities, ordered=True)
plt.figure(figsize=(10, 6))
sns.histplot(data=filtered_df, x='listed_in(city)', multiple='stack')
plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees for better readability
plt.title("Distribution of 'Pubs & bars' by City (Sorted)")
plt.show()
```


    
![png](output_64_0.png)
    



```python
# Group the data by 'location' and calculate the mean rating for each location
rating_by_location = df.groupby('location')['rating'].mean().sort_values(ascending=False)
# Display the result
print(rating_by_location)
```

    location
    Lavelle Road             4.141788
    Koramangala 3rd Block    4.020419
    St. Marks Road           4.017201
    Koramangala 5th Block    4.010215
    Church Street            3.992125
                               ...   
    Rammurthy Nagar          3.346154
    North Bangalore          3.340000
    Peenya                   3.200000
    Bommanahalli             3.190972
    Old Madras Road          3.181818
    Name: rating, Length: 92, dtype: float64



```python
# Get the top 10 locations with the highest average ratings
top_10_rating_by_location = rating_by_location.head(10)
# Display the top 10
print(top_10_rating_by_location)
```

    location
    Lavelle Road             4.141788
    Koramangala 3rd Block    4.020419
    St. Marks Road           4.017201
    Koramangala 5th Block    4.010215
    Church Street            3.992125
    Sankey Road              3.965385
    Koramangala 4th Block    3.918668
    Cunningham Road          3.901053
    Residency Road           3.865657
    Koramangala 7th Block    3.855577
    Name: rating, dtype: float64



```python
df.columns
```




    Index(['url', 'address', 'name', 'online_order', 'book_table', 'votes',
           'phone', 'location', 'rest_type', 'dish_liked', 'cuisines',
           'reviews_list', 'menu_item', 'listed_in(type)', 'listed_in(city)',
           'rating', 'Cost_For_two', 'Cuisine_Count'],
          dtype='object')




```python
import plotly.express as px

# Calculate average ratings and total votes for each location
top_10_df = df.groupby('location').agg(
    rating=('rating', 'mean'),  # Average rating
    votes=('votes', 'sum')       # Total votes
).reset_index()

# Format the votes with commas for better readability
top_10_df['votes'] = top_10_df['votes'].apply(lambda x: f"{x:,}")

# Get the top 10 locations by average rating
top_10_df = top_10_df.nlargest(10, 'rating')

# Use plotly to create an interactive bar chart with sky blue color
fig = px.bar(
    top_10_df,
    x='location',
    y='rating',
    title="Top 10 Locations by Average Rating",
    labels={'location': 'Location', 'rating': 'Average Rating', 'votes': 'Votes'},
    color_discrete_sequence=['#87CEEB'],  # Sky blue color
    hover_data=['votes']  # Include formatted votes in the hover tooltip
)

# Customize the layout
fig.update_layout(width=600, height=400)
fig.show()

```


<div>                            <div id="226b719a-4fcb-4195-90a1-acd67e2135a6" class="plotly-graph-div" style="height:400px; width:600px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("226b719a-4fcb-4195-90a1-acd67e2135a6")) {                    Plotly.newPlot(                        "226b719a-4fcb-4195-90a1-acd67e2135a6",                        [{"alignmentgroup":"True","customdata":[["505,460"],["125,159"],["266,099"],["2,214,621"],["594,979"],["6,411"],["685,156"],["287,873"],["291,535"],["495,240"]],"hovertemplate":"Location=%{x}\u003cbr\u003eAverage Rating=%{y}\u003cbr\u003eVotes=%{customdata[0]}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"#87CEEB","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","x":["Lavelle Road","Koramangala 3rd Block","St. Marks Road","Koramangala 5th Block","Church Street","Sankey Road","Koramangala 4th Block","Cunningham Road","Residency Road","Koramangala 7th Block"],"xaxis":"x","y":[4.141787941787942,4.020418848167539,4.017201166180758,4.010214818062253,3.9921245421245417,3.965384615384615,3.918668252080856,3.9010526315789473,3.8656565656565656,3.855576739752145],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Location"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Average Rating"}},"legend":{"tracegroupgap":0},"title":{"text":"Top 10 Locations by Average Rating"},"barmode":"relative","width":600,"height":400},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('226b719a-4fcb-4195-90a1-acd67e2135a6');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
import plotly.express as px

# Create a bar plot for the online_order data with percentages
fig = px.histogram(df, x='online_order', title="Distribution of Online Orders (Percentage)",
                   labels={'online_order': 'Online Order'},
                   color_discrete_sequence=['#87CEEB'],
                   histnorm='percent')  # Show percentages instead of counts

# Customize the layout and set figure size
fig.update_layout(width=600, height=400, yaxis_title="Percentage")
# Show the plot
fig.show()

```


<div>                            <div id="9e4ce2f2-5034-4e33-ab67-03fb16ef762d" class="plotly-graph-div" style="height:400px; width:600px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("9e4ce2f2-5034-4e33-ab67-03fb16ef762d")) {                    Plotly.newPlot(                        "9e4ce2f2-5034-4e33-ab67-03fb16ef762d",                        [{"alignmentgroup":"True","bingroup":"x","histnorm":"percent","hovertemplate":"Online Order=%{x}\u003cbr\u003epercent=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"#87CEEB","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"x":["Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No"],"xaxis":"x","yaxis":"y","type":"histogram"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Online Order"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Percentage"}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Online Orders (Percentage)"},"barmode":"relative","width":600,"height":400},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('9e4ce2f2-5034-4e33-ab67-03fb16ef762d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
df[df['online_order']=='No'].shape
```




    (13936, 18)




```python
fig = px.histogram(
    df,
    x='book_table',
    title="Distribution of Book Table",
    labels={'book_table': 'Book Table', 'y': 'Percentage'},
    color_discrete_sequence=['#87CEEB'],
    histnorm='percent'
)
fig.update_layout(
    width=600,
    height=400,
    yaxis_title='Percentage (%)',
    xaxis_title='Book Table',
    bargap=0.2
)


fig.update_yaxes(tickformat=".1f%")


fig.show()
```


<div>                            <div id="69219108-d5f0-4b11-9be7-da99c0cc39bb" class="plotly-graph-div" style="height:400px; width:600px;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("69219108-d5f0-4b11-9be7-da99c0cc39bb")) {                    Plotly.newPlot(                        "69219108-d5f0-4b11-9be7-da99c0cc39bb",                        [{"alignmentgroup":"True","bingroup":"x","histnorm":"percent","hovertemplate":"Book Table=%{x}\u003cbr\u003epercent=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"#87CEEB","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"x":["Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","No","No","No","No","No","Yes","Yes","Yes","Yes","No","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","No","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","No","No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","Yes","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","Yes","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","Yes","No","No","Yes","Yes","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","No","No","No","Yes","No","No","Yes","No","No","No","Yes","Yes","No","Yes","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","Yes","Yes","No","Yes","No","Yes","No","No","Yes","No","Yes","No","No","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","Yes","No","Yes","Yes","No","No","Yes","No","Yes","No","Yes","Yes","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","Yes","Yes","No","Yes","Yes","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","No","Yes","No","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","No","No","Yes","No","Yes","No","Yes","No","No","Yes","No","No","No","Yes","Yes","Yes","Yes","No","No","No","Yes","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","Yes","No","No","No","No","No","No","No","No","No","No","No","No","No","No","No","Yes","No","No","No","No","No","No","No","No","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","No","Yes","Yes","No","No","No","Yes","No","No","Yes","No","No","No","No","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","Yes","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","No","No","No","No","No","Yes","No"],"xaxis":"x","yaxis":"y","type":"histogram"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Book Table"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Percentage (%)"},"tickformat":".1f%"},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Book Table"},"barmode":"relative","width":600,"height":400,"bargap":0.2},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('69219108-d5f0-4b11-9be7-da99c0cc39bb');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


# Removing duplicate address column's values and plotting to see the branch count


```python
# Keep the last occurrence of each duplicate address
df_no_duplicates = df.drop_duplicates(subset='address', keep='last')

```


```python
df_no_duplicates.shape
```




    (8648, 18)




```python
hotel_location_counts_ = df_no_duplicates.groupby('name')['location'].nunique()

# Sort hotels by the number of locations (in descending order)
sorted_hotel_location_counts_after_removing_duplicates = hotel_location_counts_.sort_values(ascending=False)
```


```python
sorted_hotel_location_counts_after_removing_duplicates[0:10]
```




    name
    cafe coffee day      31
    domino's pizza       29
    pizza hut            27
    five star chicken    27
    kanti sweets         26
    kfc                  26
    just bake            24
    mcdonald's           22
    faasos               22
    baskin robbins       20
    Name: location, dtype: int64




```python
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")  # Set a clean background style

# Create the bar plot
ax = sns.barplot(
    x=sorted_hotel_location_counts_after_removing_duplicates[:10].index,
    y=sorted_hotel_location_counts_after_removing_duplicates[:10].values,
    palette="Blues_d"  # Choose a color palette; adjust to your preference
)

# Add labels and title
ax.set_title("Top 10 hotels with more branches", fontsize=14)
ax.set_xlabel("Hotel Location", fontsize=12)
ax.set_ylabel("Count", fontsize=12)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha="right")

# Add data labels on top of each bar
for p in ax.patches:
    ax.annotate(
        format(int(p.get_height()), ","),
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha="center", va="center",
        xytext=(0, 5), textcoords="offset points"
    )

# Remove unnecessary spines for a cleaner look
sns.despine()

plt.show()
```


    
![png](output_77_0.png)
    



```python
df_no_duplicates.sort_values(by='name', ascending=True)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49511</th>
      <td>https://www.zomato.com/bangalore/feeltheroll-b...</td>
      <td>Opposite Mantri Commercio, Outer Ring Road, De...</td>
      <td>#feeltheroll</td>
      <td>No</td>
      <td>No</td>
      <td>7</td>
      <td>+91 9108342079\n+91 9886117901</td>
      <td>Bellandur</td>
      <td>Quick Bites</td>
      <td>NaN</td>
      <td>fast food</td>
      <td>[('Rated 5.0', "RATED\n  Had an egg chicken ro...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Sarjapur Road</td>
      <td>3.4</td>
      <td>200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36386</th>
      <td>https://www.zomato.com/bangalore/l-81-cafe-hsr...</td>
      <td>Sector 6, HSR Layout, HSR</td>
      <td>#l-81 cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>48</td>
      <td>+91 9986210891</td>
      <td>HSR</td>
      <td>Quick Bites</td>
      <td>Burgers</td>
      <td>fast food, beverages</td>
      <td>[('Rated 4.0', 'RATED\n  This little cafe is s...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Koramangala 7th Block</td>
      <td>3.9</td>
      <td>400</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1967</th>
      <td>https://www.zomato.com/bangalore/refuel-banner...</td>
      <td>7, Ground Floor, RR Commercial Complex, Akshay...</td>
      <td>#refuel</td>
      <td>Yes</td>
      <td>No</td>
      <td>37</td>
      <td>+91 8971227222</td>
      <td>Bannerghatta Road</td>
      <td>Cafe</td>
      <td>Thick Shakes, Sandwiches, Pasta, Mocktails</td>
      <td>cafe, beverages</td>
      <td>[('Rated 3.0', 'RATED\n  We ordered for Schezw...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Bannerghatta Road</td>
      <td>3.7</td>
      <td>400</td>
      <td>2</td>
    </tr>
    <tr>
      <th>35295</th>
      <td>https://www.zomato.com/bangalore/1000-b-c-kora...</td>
      <td>16, 17th A Main, Koramangala 5th Block, Bangalore</td>
      <td>1000 b.c</td>
      <td>Yes</td>
      <td>No</td>
      <td>49</td>
      <td>+91 9620946663</td>
      <td>Koramangala 5th Block</td>
      <td>Quick Bites</td>
      <td>Shawarma, Sandwiches</td>
      <td>arabian, sandwich, rolls, burger</td>
      <td>[('Rated 1.0', "RATED\n  Ordered a chicken sub...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Koramangala 7th Block</td>
      <td>3.2</td>
      <td>300</td>
      <td>4</td>
    </tr>
    <tr>
      <th>23738</th>
      <td>https://www.zomato.com/bangalore/100%C2%B0c-bt...</td>
      <td>688, Thanish Corner, 7th Main, 10th Cross, 2nd...</td>
      <td>100ãâãâãâãâãâãâãâãâ°c</td>
      <td>No</td>
      <td>No</td>
      <td>41</td>
      <td>+91 9535433735</td>
      <td>BTM</td>
      <td>Casual Dining</td>
      <td>Chicken Biryani</td>
      <td>biryani, north indian</td>
      <td>[('Rated 1.0', 'RATED\n  They are frod they ha...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>JP Nagar</td>
      <td>3.7</td>
      <td>450</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>48018</th>
      <td>https://www.zomato.com/bangalore/zoroy-luxury-...</td>
      <td>44, Near K C Das, Off Brigade Road, Church Str...</td>
      <td>zoroy luxury chocolate</td>
      <td>Yes</td>
      <td>No</td>
      <td>68</td>
      <td>080 41126811\n+91 8880066000</td>
      <td>Church Street</td>
      <td>Dessert Parlor</td>
      <td>Hot Chocolate</td>
      <td>desserts</td>
      <td>[('Rated 4.0', 'RATED\n  A good shop for handm...</td>
      <td>[]</td>
      <td>Desserts</td>
      <td>Residency Road</td>
      <td>4.0</td>
      <td>250</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39940</th>
      <td>https://www.zomato.com/bangalore/zus-doner-keb...</td>
      <td>No 214, opp nandini hotel, 80 ft main road, RT...</td>
      <td>zu's doner kebaps</td>
      <td>No</td>
      <td>No</td>
      <td>33</td>
      <td>NaN</td>
      <td>RT Nagar</td>
      <td>Takeaway, Delivery</td>
      <td>NaN</td>
      <td>turkish, fast food, biryani, chinese</td>
      <td>[('Rated 5.0', ''), ('Rated 2.0', ''), ('Rated...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Malleshwaram</td>
      <td>3.7</td>
      <td>350</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26027</th>
      <td>https://www.zomato.com/bangalore/zus-doner-keb...</td>
      <td>44, Shop 1, 5th Cross, 5th Main, KEB Main Road...</td>
      <td>zu's doner kebaps</td>
      <td>Yes</td>
      <td>No</td>
      <td>22</td>
      <td>+91 9611900777</td>
      <td>Kammanahalli</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>turkish, fast food</td>
      <td>[('Rated 1.0', 'RATED\n  for came 45 minutes l...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Kammanahalli</td>
      <td>3.6</td>
      <td>350</td>
      <td>2</td>
    </tr>
    <tr>
      <th>26169</th>
      <td>https://www.zomato.com/bangalore/zyara-hbr-lay...</td>
      <td>46, 80 Feet Road, Opposite HP Petrol Pump, HBR...</td>
      <td>zyara</td>
      <td>Yes</td>
      <td>No</td>
      <td>191</td>
      <td>+91 9148444441\n+91 9148444442</td>
      <td>HBR Layout</td>
      <td>Casual Dining</td>
      <td>Shawarma, Chicken Grill, Tandoori Chicken, Chi...</td>
      <td>north indian, mughlai, chinese</td>
      <td>[('Rated 3.0', 'RATED\n  When I visited this p...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Kammanahalli</td>
      <td>3.8</td>
      <td>650</td>
      <td>3</td>
    </tr>
    <tr>
      <th>23285</th>
      <td>https://www.zomato.com/zyksha-food-truck?conte...</td>
      <td>JP Nagar, Bengaluru</td>
      <td>zyksha</td>
      <td>No</td>
      <td>No</td>
      <td>9</td>
      <td>NaN</td>
      <td>South Bangalore</td>
      <td>Food Truck</td>
      <td>NaN</td>
      <td>fast food</td>
      <td>[('Rated 4.5', 'RATED\n  "Burp!" ... That Crun...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>JP Nagar</td>
      <td>3.4</td>
      <td>200</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>8648 rows × 18 columns</p>
</div>




```python
hotel_counts = df_no_duplicates.groupby('location')['name'].nunique()
hotel_counts
```




    location
    BTM                  475
    Banashankari         200
    Banaswadi            135
    Bannerghatta Road    312
    Basavanagudi         156
                        ... 
    West Bangalore         1
    Whitefield           525
    Wilson Garden         30
    Yelahanka              2
    Yeshwantpur           59
    Name: name, Length: 91, dtype: int64




```python

# Get top 10 locations with the most unique hotels
top_10_locations = hotel_counts.nlargest(10)

# Plotting
plt.figure(figsize=(10, 6))
top_10_locations.plot(kind='bar', color='skyblue')

# Adding labels and title
plt.title('Top 10 Locations with Most Unique Hotels', fontsize=16)
plt.xlabel('Location', fontsize=12)
plt.ylabel('Number of Unique Hotels', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
```


    
![png](output_80_0.png)
    



```python
hotel_counts.sum()
```




    np.int64(8413)




```python
df_no_duplicates.name.nunique()
```




    6103




```python
df_no_duplicates['listed_in(city)'].nunique()
```




    30




```python
df_no_duplicates['location'].nunique()
```




    91



Book table vs. area


```python
# Filter for rows where 'book_table' is 'Yes' and group by 'location' to get the count
yes_book_table_by_location = df_no_duplicates[df_no_duplicates['book_table'] == 'Yes'].groupby('listed_in(city)').size()
# Alternatively, if you want a DataFrame format with counts
yes_book_table_by_location = df_no_duplicates[df_no_duplicates['book_table'] == 'Yes'].groupby('listed_in(city)').agg({'book_table': 'count'}).rename(columns={'book_table': 'count'})
```


```python
# Sorting by the 'count' column if it’s a DataFrame
yes_book_table_by_location = yes_book_table_by_location.sort_values(by='count', ascending=False)
```


```python
yes_book_table_by_location[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>listed_in(city)</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Koramangala 7th Block</th>
      <td>142</td>
    </tr>
    <tr>
      <th>Residency Road</th>
      <td>117</td>
    </tr>
    <tr>
      <th>Old Airport Road</th>
      <td>88</td>
    </tr>
    <tr>
      <th>Whitefield</th>
      <td>77</td>
    </tr>
    <tr>
      <th>JP Nagar</th>
      <td>66</td>
    </tr>
    <tr>
      <th>Kammanahalli</th>
      <td>52</td>
    </tr>
    <tr>
      <th>Marathahalli</th>
      <td>47</td>
    </tr>
    <tr>
      <th>Sarjapur Road</th>
      <td>44</td>
    </tr>
    <tr>
      <th>Rajajinagar</th>
      <td>36</td>
    </tr>
    <tr>
      <th>Electronic City</th>
      <td>33</td>
    </tr>
  </tbody>
</table>
</div>




```python
# filtered_df = df_no_duplicates[(df_no_duplicates['book_table'] == 'Yes') & (df_no_duplicates['listed_in(city)'] == 'Koramangala 7th Block')]
# filtered_df.sort_values(by='name', ascending=True)
```


```python
filtered_df.shape
```




    (632, 18)



# Try with duplicate data


```python
# Filter for rows where 'book_table' is 'Yes' and group by 'location' to get the count
yes_book_table_by_location = df[df['book_table'] == 'Yes'].groupby('listed_in(city)').size()
# Alternatively, if you want a DataFrame format with counts
yes_book_table_by_location = df[df['book_table'] == 'Yes'].groupby('listed_in(city)').agg({'book_table': 'count'}).rename(columns={'book_table': 'count'})
```


```python
# Sorting by the 'count' column if it’s a DataFrame
yes_book_table_by_location = yes_book_table_by_location.sort_values(by='count', ascending=False)
```


```python
yes_book_table_by_location[0:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>listed_in(city)</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Koramangala 7th Block</th>
      <td>373</td>
    </tr>
    <tr>
      <th>BTM</th>
      <td>364</td>
    </tr>
    <tr>
      <th>Church Street</th>
      <td>362</td>
    </tr>
    <tr>
      <th>MG Road</th>
      <td>356</td>
    </tr>
    <tr>
      <th>Koramangala 4th Block</th>
      <td>355</td>
    </tr>
    <tr>
      <th>Koramangala 5th Block</th>
      <td>352</td>
    </tr>
    <tr>
      <th>Brigade Road</th>
      <td>349</td>
    </tr>
    <tr>
      <th>Lavelle Road</th>
      <td>327</td>
    </tr>
    <tr>
      <th>Koramangala 6th Block</th>
      <td>321</td>
    </tr>
    <tr>
      <th>Indiranagar</th>
      <td>318</td>
    </tr>
  </tbody>
</table>
</div>




```python
filtered_df = df[(df['book_table'] == 'Yes') & (df['listed_in(city)'] == 'Koramangala 7th Block')]
```


```python
filtered_df.sort_values(by='name', ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34752</th>
      <td>https://www.zomato.com/bangalore/12th-main-gra...</td>
      <td>Grand Mercure, 12th Main, 3rd Block, Koramanga...</td>
      <td>12th main - grand mercure</td>
      <td>No</td>
      <td>Yes</td>
      <td>354</td>
      <td>080 45121638\n080 45121212</td>
      <td>Koramangala 3rd Block</td>
      <td>Fine Dining</td>
      <td>Halwa, Waffles, Chaat, Pasta, Coffee, Creme Br...</td>
      <td>european, asian</td>
      <td>[('Rated 2.0', 'RATED\n  Went here recently fo...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Koramangala 7th Block</td>
      <td>4.1</td>
      <td>2000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36830</th>
      <td>https://www.zomato.com/bangalore/12th-main-gra...</td>
      <td>Grand Mercure, 12th Main, 3rd Block, Koramanga...</td>
      <td>12th main - grand mercure</td>
      <td>No</td>
      <td>Yes</td>
      <td>357</td>
      <td>080 45121638\n080 45121212</td>
      <td>Koramangala 3rd Block</td>
      <td>Fine Dining</td>
      <td>Halwa, Waffles, Chaat, Pasta, Coffee, Creme Br...</td>
      <td>european, asian</td>
      <td>[('Rated 5.0', 'RATED\n  Like I always say in ...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Koramangala 7th Block</td>
      <td>4.1</td>
      <td>2000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>36708</th>
      <td>https://www.zomato.com/bangalore/1522-the-pub-...</td>
      <td>3, 80 Feet Road, 4-C Block, Koramangala 4th Bl...</td>
      <td>1522 - the pub</td>
      <td>No</td>
      <td>Yes</td>
      <td>1746</td>
      <td>+91 9980985527\n+91 9481451597</td>
      <td>Koramangala 4th Block</td>
      <td>Pub</td>
      <td>Cocktails, Devils Chicken, Beer, Tandoori Chic...</td>
      <td>chinese, continental, north indian</td>
      <td>[('Rated 4.0', 'RATED\n  The pasta was amazing...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Koramangala 7th Block</td>
      <td>4.2</td>
      <td>1400</td>
      <td>3</td>
    </tr>
    <tr>
      <th>35984</th>
      <td>https://www.zomato.com/bangalore/1522-the-pub-...</td>
      <td>3, 80 Feet Road, 4-C Block, Koramangala 4th Bl...</td>
      <td>1522 - the pub</td>
      <td>No</td>
      <td>Yes</td>
      <td>1745</td>
      <td>+91 9980985527\n+91 9481451597</td>
      <td>Koramangala 4th Block</td>
      <td>Pub</td>
      <td>Cocktails, Devils Chicken, Beer, Tandoori Chic...</td>
      <td>chinese, continental, north indian</td>
      <td>[('Rated 4.0', 'RATED\n  The pasta was amazing...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>Koramangala 7th Block</td>
      <td>4.2</td>
      <td>1400</td>
      <td>3</td>
    </tr>
    <tr>
      <th>37561</th>
      <td>https://www.zomato.com/bangalore/1522-the-pub-...</td>
      <td>3, 80 Feet Road, 4-C Block, Koramangala 4th Bl...</td>
      <td>1522 - the pub</td>
      <td>No</td>
      <td>Yes</td>
      <td>1746</td>
      <td>080 49652895</td>
      <td>Koramangala 4th Block</td>
      <td>Pub</td>
      <td>Cocktails, Devils Chicken, Beer, Tandoori Chic...</td>
      <td>chinese, continental, north indian</td>
      <td>[('Rated 4.0', 'RATED\n  The pasta was amazing...</td>
      <td>[]</td>
      <td>Drinks &amp; nightlife</td>
      <td>Koramangala 7th Block</td>
      <td>4.2</td>
      <td>1400</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>37567</th>
      <td>https://www.zomato.com/bangalore/whats-in-a-na...</td>
      <td>146, Next to William Penn Showroom, Koramangal...</td>
      <td>what's in a name</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1798</td>
      <td>080 49652465</td>
      <td>Koramangala 5th Block</td>
      <td>Pub</td>
      <td>Cocktails, Chicken Nachos, Pizza, Peri Peri Fr...</td>
      <td>finger food, asian, european, italian</td>
      <td>[('Rated 3.0', 'RATED\n  We went over for a qu...</td>
      <td>['Chilli Chicken', 'Beef Steak with Mash and V...</td>
      <td>Drinks &amp; nightlife</td>
      <td>Koramangala 7th Block</td>
      <td>4.1</td>
      <td>1200</td>
      <td>4</td>
    </tr>
    <tr>
      <th>36670</th>
      <td>https://www.zomato.com/bangalore/xoox-brewmill...</td>
      <td>8, Koramanagala Industrial Layout, Near HDFC B...</td>
      <td>xoox brewmill</td>
      <td>No</td>
      <td>Yes</td>
      <td>1592</td>
      <td>080 49652469</td>
      <td>Koramangala 5th Block</td>
      <td>Microbrewery, Casual Dining</td>
      <td>Cocktails, Apple Cider, Craft Beer, Burgers, S...</td>
      <td>modern indian, european, asian</td>
      <td>[('Rated 5.0', "RATED\n  It was ABSOLUTELY LOV...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>Koramangala 7th Block</td>
      <td>4.4</td>
      <td>2000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>37540</th>
      <td>https://www.zomato.com/bangalore/xoox-brewmill...</td>
      <td>8, Koramanagala Industrial Layout, Near HDFC B...</td>
      <td>xoox brewmill</td>
      <td>No</td>
      <td>Yes</td>
      <td>1592</td>
      <td>080 49652469</td>
      <td>Koramangala 5th Block</td>
      <td>Microbrewery, Casual Dining</td>
      <td>Cocktails, Apple Cider, Craft Beer, Burgers, S...</td>
      <td>modern indian, european, asian</td>
      <td>[('Rated 5.0', "RATED\n  It was ABSOLUTELY LOV...</td>
      <td>[]</td>
      <td>Drinks &amp; nightlife</td>
      <td>Koramangala 7th Block</td>
      <td>4.4</td>
      <td>2000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>34726</th>
      <td>https://www.zomato.com/bangalore/xoox-brewmill...</td>
      <td>8, Koramanagala Industrial Layout, Near HDFC B...</td>
      <td>xoox brewmill</td>
      <td>No</td>
      <td>Yes</td>
      <td>1577</td>
      <td>080 49652469</td>
      <td>Koramangala 5th Block</td>
      <td>Microbrewery, Casual Dining</td>
      <td>Cocktails, Apple Cider, Craft Beer, Burgers, S...</td>
      <td>modern indian, european, asian</td>
      <td>[('Rated 3.0', "RATED\n  This is a really pret...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Koramangala 7th Block</td>
      <td>4.4</td>
      <td>2000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>34928</th>
      <td>https://www.zomato.com/bangalore/zero-mile-pun...</td>
      <td>5th Floor, L167, Outer Ring Road, Service Lane...</td>
      <td>zero mile punjab</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1342</td>
      <td>+91 9986785586</td>
      <td>HSR</td>
      <td>Casual Dining</td>
      <td>Lassi, Chicken Curry, Chicken Malai Tikka, But...</td>
      <td>north indian, mughlai</td>
      <td>[('Rated 2.0', 'RATED\n  We went there for din...</td>
      <td>['Paneer Makahnwala', 'Paneer Methi Malai', 'P...</td>
      <td>Delivery</td>
      <td>Koramangala 7th Block</td>
      <td>4.1</td>
      <td>800</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>373 rows × 18 columns</p>
</div>



Checking whether the mismatched filtration works fine



```python
# duplicates=filtered_df = pd.DataFrame({'name':['a','a','b'],'book_table':['Yes','No','Yes']})
# mismatched_book_table = duplicates.groupby('name').filter(lambda x: x['book_table'].nunique() > 1)

# mismatched_book_table
```


```python
# Find rows with duplicate 'name' values
duplicates = filtered_df[filtered_df.duplicated(subset=['name'], keep=False)]

# Group by 'name' and filter groups where 'book_table' values are not the same
mismatched_book_table = duplicates.groupby('name').filter(lambda x: x['book_table'].nunique() > 1)

mismatched_book_table

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
print("Dimension of the dataframe named 'df' ",df.shape)
print("Dimension of the dataframe named 'df_no_duplicates.shape'",df_no_duplicates.shape)
```

    Dimension of the dataframe named 'df'  (40764, 18)
    Dimension of the dataframe named 'df_no_duplicates.shape' (8648, 18)



```python
# !pip install ipywidgets
```


```python
from IPython.display import display
import ipywidgets as widgets
```


```python
# Dropdown widget for location input
location_input = widgets.Combobox(
    placeholder="Type a location...",
    options=df['listed_in(city)'].unique().tolist(),
    description="Location:",
    ensure_option=False,
    continuous_update=True
)

# Output widget to display results
output = widgets.Output()

def display_bookable_shops(change):
    """Callback function to filter and display results."""
    output.clear_output()
    location = change['new']  # Get user input
    if location:
        filtered_df = df[(df['book_table'] == 'Yes') & (df['listed_in(city)'].str.contains(location, case=False))]
        with output:
            if not filtered_df.empty:
                display(filtered_df[['name', 'listed_in(city)']].sort_values(by='name'))
            else:
                print("No table bookable shops found for the entered location.")

# Attach the callback to the input widget
location_input.observe(display_bookable_shops, names='value')

# Display the input widget and output
display(location_input, output)
```


    Combobox(value='', description='Location:', options=('Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bel…



    Output()



```python
table_bookable = input("Enter the location to see the table bookable food shops:")
filter=(df['book_table'] == 'Yes') & (df['listed_in(city)'] == table_bookable)
filtered_df = df[filter]
filtered_df.sort_values(by='name', ascending=True)
```

    Enter the location to see the table bookable food shops: BTM





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8291</th>
      <td>https://www.zomato.com/bangalore/12th-main-gra...</td>
      <td>Grand Mercure, 12th Main, 3rd Block, Koramanga...</td>
      <td>12th main - grand mercure</td>
      <td>No</td>
      <td>Yes</td>
      <td>353</td>
      <td>080 45121638\r\n080 45121212</td>
      <td>Koramangala 3rd Block</td>
      <td>Fine Dining</td>
      <td>Halwa, Waffles, Chaat, Pasta, Coffee, Creme Br...</td>
      <td>european, asian</td>
      <td>[('Rated 2.0', 'RATED\n  Went here recently fo...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>BTM</td>
      <td>4.1</td>
      <td>2000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11064</th>
      <td>https://www.zomato.com/bangalore/12th-main-gra...</td>
      <td>Grand Mercure, 12th Main, 3rd Block, Koramanga...</td>
      <td>12th main - grand mercure</td>
      <td>No</td>
      <td>Yes</td>
      <td>353</td>
      <td>080 45121638\r\r\n080 45121212</td>
      <td>Koramangala 3rd Block</td>
      <td>Fine Dining</td>
      <td>Halwa, Waffles, Chaat, Pasta, Coffee, Creme Br...</td>
      <td>european, asian</td>
      <td>[('Rated 2.0', 'RATED\n  Went here recently fo...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>BTM</td>
      <td>4.1</td>
      <td>2000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8949</th>
      <td>https://www.zomato.com/bangalore/154-breakfast...</td>
      <td>154, 8th Main Road, 3rd Block, Koramangala 3rd...</td>
      <td>154 breakfast club</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1509</td>
      <td>080 25533133</td>
      <td>Koramangala 3rd Block</td>
      <td>Cafe</td>
      <td>Pancakes, Waffles, Bbq Sandwich, Mushroom Rago...</td>
      <td>cafe, continental</td>
      <td>[('Rated 4.0', "RATED\n  Good place for breakf...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>BTM</td>
      <td>4.0</td>
      <td>900</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10957</th>
      <td>https://www.zomato.com/bangalore/154-breakfast...</td>
      <td>154, 8th Main Road, 3rd Block, Koramangala 3rd...</td>
      <td>154 breakfast club</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1509</td>
      <td>080 25533133</td>
      <td>Koramangala 3rd Block</td>
      <td>Cafe</td>
      <td>Pancakes, Waffles, Bbq Sandwich, Mushroom Rago...</td>
      <td>cafe, continental</td>
      <td>[('Rated 4.0', "RATED\n  Good place for breakf...</td>
      <td>[]</td>
      <td>Dine-out</td>
      <td>BTM</td>
      <td>4.0</td>
      <td>900</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8347</th>
      <td>https://www.zomato.com/bangalore/154-breakfast...</td>
      <td>154, 8th Main Road, 3rd Block, Koramangala 3rd...</td>
      <td>154 breakfast club</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1509</td>
      <td>080 25533133</td>
      <td>Koramangala 3rd Block</td>
      <td>Cafe</td>
      <td>Pancakes, Waffles, Bbq Sandwich, Mushroom Rago...</td>
      <td>cafe, continental</td>
      <td>[('Rated 4.0', "RATED\n  Good place for breakf...</td>
      <td>[]</td>
      <td>Cafes</td>
      <td>BTM</td>
      <td>4.0</td>
      <td>900</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8269</th>
      <td>https://www.zomato.com/bangalore/xoox-brewmill...</td>
      <td>8, Koramanagala Industrial Layout, Near HDFC B...</td>
      <td>xoox brewmill</td>
      <td>No</td>
      <td>Yes</td>
      <td>1533</td>
      <td>080 49652469</td>
      <td>Koramangala 5th Block</td>
      <td>Microbrewery, Casual Dining</td>
      <td>Cocktails, Apple Cider, Craft Beer, Burgers, S...</td>
      <td>modern indian, european, asian</td>
      <td>[('Rated 5.0', "RATED\n  Highly recommend the ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>BTM</td>
      <td>4.4</td>
      <td>2000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11452</th>
      <td>https://www.zomato.com/bangalore/xoox-brewmill...</td>
      <td>8, Koramanagala Industrial Layout, Near HDFC B...</td>
      <td>xoox brewmill</td>
      <td>No</td>
      <td>Yes</td>
      <td>1533</td>
      <td>080 49652469</td>
      <td>Koramangala 5th Block</td>
      <td>Microbrewery, Casual Dining</td>
      <td>Cocktails, Apple Cider, Craft Beer, Burgers, S...</td>
      <td>modern indian, european, asian</td>
      <td>[]</td>
      <td>[]</td>
      <td>Drinks &amp; nightlife</td>
      <td>BTM</td>
      <td>4.4</td>
      <td>2000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9048</th>
      <td>https://www.zomato.com/bangalore/zaitoon-jp-na...</td>
      <td>21, 24th Main Road, 6th Phase, JP Nagar, Banga...</td>
      <td>zaitoon</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1543</td>
      <td>080 41477977\r\r\n080 41488988</td>
      <td>JP Nagar</td>
      <td>Casual Dining</td>
      <td>Hummus Falafel, Gulab Jamun, Shawarma, Bbq Chi...</td>
      <td>bbq, arabian, chinese, north indian, desserts</td>
      <td>[('Rated 5.0', "RATED\n  Bedridden and after s...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>BTM</td>
      <td>4.1</td>
      <td>1000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10966</th>
      <td>https://www.zomato.com/bangalore/zero-mile-pun...</td>
      <td>5th Floor, L167, Outer Ring Road, Service Lane...</td>
      <td>zero mile punjab</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1336</td>
      <td>+91 9986785586</td>
      <td>HSR</td>
      <td>Casual Dining</td>
      <td>Lassi, Chicken Curry, Chicken Malai Tikka, But...</td>
      <td>north indian, mughlai</td>
      <td>[('Rated 4.0', 'RATED\n  Came here last week. ...</td>
      <td>['Lassi Bhar Ke [400 ml]', 'Lassi Halkee Phulk...</td>
      <td>Dine-out</td>
      <td>BTM</td>
      <td>4.1</td>
      <td>800</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8903</th>
      <td>https://www.zomato.com/bangalore/zero-mile-pun...</td>
      <td>5th Floor, L167, Outer Ring Road, Service Lane...</td>
      <td>zero mile punjab</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1336</td>
      <td>+91 9986785586</td>
      <td>HSR</td>
      <td>Casual Dining</td>
      <td>Lassi, Chicken Curry, Chicken Malai Tikka, But...</td>
      <td>north indian, mughlai</td>
      <td>[('Rated 4.0', 'RATED\n  Came here last week. ...</td>
      <td>[]</td>
      <td>Delivery</td>
      <td>BTM</td>
      <td>4.1</td>
      <td>800</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 18 columns</p>
</div>



# Preparing Data Set for Recommendation system

## Cleaning Reviews feature , First taking 10 rows , to check wether the cleaning logic works goo0d.


```python
# Slice the first 10 rows of the DataFrame
df_subset = df.iloc[:10].copy()

# Assign 'reviews_list' to the new column
df_subset['reviews_list test'] = df_subset['reviews_list']

# Check the shape
print(df_subset.shape)  # Should be (10, <number_of_columns + 1>)

```

    (10, 19)



```python
df_subset['reviews_list test'][0]

```




    '[(\'Rated 4.0\', \'RATED\\n  A beautiful place to dine in.The interiors take you back to the Mughal era. The lightings are just perfect.We went there on the occasion of Christmas and so they had only limited items available. But the taste and service was not compromised at all.The only complaint is that the breads could have been better.Would surely like to come here again.\'), (\'Rated 4.0\', \'RATED\\n  I was here for dinner with my family on a weekday. The restaurant was completely empty. Ambience is good with some good old hindi music. Seating arrangement are good too. We ordered masala papad, panner and baby corn starters, lemon and corrionder soup, butter roti, olive and chilli paratha. Food was fresh and good, service is good too. Good for family hangout.\\nCheers\'), (\'Rated 2.0\', \'RATED\\n  Its a restaurant near to Banashankari BDA. Me along with few of my office friends visited to have buffet but unfortunately they only provide veg buffet. On inquiring they said this place is mostly visited by vegetarians. Anyways we ordered ala carte items which took ages to come. Food was ok ok. Definitely not visiting anymore.\'), (\'Rated 4.0\', \'RATED\\n  We went here on a weekend and one of us had the buffet while two of us took Ala Carte. Firstly the ambience and service of this place is great! The buffet had a lot of items and the good was good. We had a Pumpkin Halwa intm the dessert which was amazing. Must try! The kulchas are great here. Cheers!\'), (\'Rated 5.0\', \'RATED\\n  The best thing about the place is itÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x92s ambiance. Second best thing was yummy ? food. We try buffet and buffet food was not disappointed us.\\nTest ?. ?? ?? ?? ?? ??\\nQuality ?. ??????????.\\nService: Staff was very professional and friendly.\\n\\nOverall experience was excellent.\\n\\nsubirmajumder85.wixsite.com\'), (\'Rated 5.0\', \'RATED\\n  Great food and pleasant ambience. Expensive but Coll place to chill and relax......\\n\\nService is really very very good and friendly staff...\\n\\nFood : 5/5\\nService : 5/5\\nAmbience :5/5\\nOverall :5/5\'), (\'Rated 4.0\', \'RATED\\n  Good ambience with tasty food.\\nCheese chilli paratha with Bhutta palak methi curry is a good combo.\\nLemon Chicken in the starters is a must try item.\\nEgg fried rice was also quite tasty.\\nIn the mocktails, recommend "Alice in Junoon". Do not miss it.\'), (\'Rated 4.0\', \'RATED\\n  You canÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x92t go wrong with Jalsa. Never been a fan of their buffet and thus always order alacarteÃ\x83\\x83Ã\x82\\x83Ã\x83\\x82Ã\x82\\x82Ã\x83\\x83Ã\x82\\x82Ã\x83\\x82Ã\x82\\x92. Service at times can be on the slower side but food is worth the wait.\'), (\'Rated 5.0\', \'RATED\\n  Overdelighted by the service and food provided at this place. A royal and ethnic atmosphere builds a strong essence of being in India and also the quality and taste of food is truly authentic. I would totally recommend to visit this place once.\'), (\'Rated 4.0\', \'RATED\\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\\n\\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome\'), (\'Rated 4.0\', \'RATED\\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\\n\\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome\'), (\'Rated 4.0\', \'RATED\\n  The place is nice and comfortable. Food wise all jalea outlets maintain a good standard. The soya chaap was a standout dish. Clearly one of trademark dish as per me and a must try.\\n\\nThe only concern is the parking. It very congested and limited to just 5cars. The basement parking is very steep and makes it cumbersome\')]'




```python
# Function to clean the review text
def clean_review(review_text):
    # Remove the 'rated' part at the beginning and any extra spaces
    review_text = re.sub(r'^rated[^\w]*', '', review_text)

    # Remove newline characters
    review_text = review_text.replace("\n", " ")

    # Fix encoding issues (you can add more replacements for any problematic encodings)
    review_text = review_text.replace('ã\x83', '').replace('ã\x82', '').replace('ã\x82', '')

    # Remove extra spaces
    review_text = ' '.join(review_text.split())

    return review_text

# Function to clean the nested tuple inside the list
def clean_nested_tuples(text):
    try:
        # Use ast.literal_eval to safely evaluate the string into a Python object
        list_of_tuples = ast.literal_eval(text)  # Safely converts string to a list of tuples

        # Clean each tuple (both rating and review text)
        cleaned_list = [(rating, clean_review(review)) for rating, review in list_of_tuples]
        return cleaned_list
    except Exception as e:
        print(f"Error while parsing or cleaning the text: {e}")
        return []

# Apply the cleaning function to each row in the 'reviews_list' column
df_subset['reviews_list test'] = df_subset['reviews_list test'].apply(clean_nested_tuples)
```


```python
# Step 1: Extract the 'reviews_list test' column
df_subset = df_subset['reviews_list test']

# Step 2: Explode the list of tuples into rows (each tuple is a separate row)
df_subset = df_subset.explode().reset_index(drop=True)

```


```python
df_subset
```




    0      (Rated 4.0, RATED A beautiful place to dine in...
    1      (Rated 4.0, RATED I was here for dinner with m...
    2      (Rated 2.0, RATED Its a restaurant near to Ban...
    3      (Rated 4.0, RATED We went here on a weekend an...
    4      (Rated 5.0, RATED The best thing about the pla...
                                 ...                        
    213    (Rated 5.0, RATED Being in banashankari, this ...
    214    (Rated 4.0, RATED Our always go to place when ...
    215    (Rated 4.0, RATED Yummy saucy fries!! Loved ev...
    216    (Rated 4.0, RATED Value for money everything t...
    217    (Rated 3.0, RATED It's a busy eatery in Banash...
    Name: reviews_list test, Length: 218, dtype: object




```python
recom_df = df
recom_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st Main Road, 2nd Stage, Banashankari, ...</td>
      <td>jalsa</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>775</td>
      <td>080 42297555\r\n+91 9743772233</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Pasta, Lunch Buffet, Masala Papad, Paneer Laja...</td>
      <td>north indian, mughlai, chinese</td>
      <td>[('Rated 4.0', 'RATED\n  A beautiful place to ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd Floor, 80 Feet Road, Near Big Bazaar, 6th ...</td>
      <td>spice elephant</td>
      <td>Yes</td>
      <td>No</td>
      <td>787</td>
      <td>080 41714161</td>
      <td>Banashankari</td>
      <td>Casual Dining</td>
      <td>Momos, Lunch Buffet, Chocolate Nirvana, Thai G...</td>
      <td>chinese, north indian, thai</td>
      <td>[('Rated 4.0', 'RATED\n  Had been here for din...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/SanchurroBangalore?cont...</td>
      <td>1112, Next to KIMS Medical College, 17th Cross...</td>
      <td>san churro cafe</td>
      <td>Yes</td>
      <td>No</td>
      <td>918</td>
      <td>+91 9663487993</td>
      <td>Banashankari</td>
      <td>Cafe, Casual Dining</td>
      <td>Churros, Cannelloni, Minestrone Soup, Hot Choc...</td>
      <td>cafe, mexican, italian</td>
      <td>[('Rated 3.0', "RATED\n  Ambience is not that ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st Floor, Annakuteera, 3rd Stage, Banashankar...</td>
      <td>addhuri udupi bhojana</td>
      <td>No</td>
      <td>No</td>
      <td>88</td>
      <td>+91 9620009302</td>
      <td>Banashankari</td>
      <td>Quick Bites</td>
      <td>Masala Dosa</td>
      <td>south indian, north indian</td>
      <td>[('Rated 4.0', "RATED\n  Great food and proper...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.7</td>
      <td>300</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd Floor, Lakshmi Associates, Gandhi Baza...</td>
      <td>grand village</td>
      <td>No</td>
      <td>No</td>
      <td>166</td>
      <td>+91 8026612447\r\n+91 9901210005</td>
      <td>Basavanagudi</td>
      <td>Casual Dining</td>
      <td>Panipuri, Gol Gappe</td>
      <td>north indian, rajasthani</td>
      <td>[('Rated 4.0', 'RATED\n  Very good restaurant ...</td>
      <td>[]</td>
      <td>Buffet</td>
      <td>Banashankari</td>
      <td>3.8</td>
      <td>600</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51709</th>
      <td>https://www.zomato.com/bangalore/the-farm-hous...</td>
      <td>136, SAP Labs India, KIADB Export Promotion In...</td>
      <td>the farm house bar n grill</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>+91 9980121279\n+91 9900240646</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>NaN</td>
      <td>north indian, continental</td>
      <td>[('Rated 4.0', 'RATED\n  Ambience- Big and spa...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.7</td>
      <td>800</td>
      <td>2</td>
    </tr>
    <tr>
      <th>51711</th>
      <td>https://www.zomato.com/bangalore/bhagini-2-whi...</td>
      <td>139/C1, Next To GR Tech Park, Pattandur Agraha...</td>
      <td>bhagini</td>
      <td>No</td>
      <td>No</td>
      <td>81</td>
      <td>080 65951222</td>
      <td>Whitefield</td>
      <td>Casual Dining, Bar</td>
      <td>Biryani, Andhra Meal</td>
      <td>andhra, south indian, chinese, north indian</td>
      <td>[('Rated 4.0', 'RATED\n  A fine place to chill...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>2.5</td>
      <td>800</td>
      <td>4</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>Four Points by Sheraton Bengaluru, 43/3, White...</td>
      <td>best brews - four points by sheraton bengaluru...</td>
      <td>No</td>
      <td>No</td>
      <td>27</td>
      <td>080 40301477</td>
      <td>Whitefield</td>
      <td>Bar</td>
      <td>NaN</td>
      <td>continental</td>
      <td>[('Rated 5.0', "RATED\n  Food and service are ...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.6</td>
      <td>1500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>Sheraton Grand Bengaluru Whitefield Hotel &amp; Co...</td>
      <td>chime - sheraton grand bengaluru whitefield ho...</td>
      <td>No</td>
      <td>Yes</td>
      <td>236</td>
      <td>080 49652769</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar</td>
      <td>Cocktails, Pizza, Buttermilk</td>
      <td>finger food</td>
      <td>[('Rated 4.0', 'RATED\n  Nice and friendly pla...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>4.3</td>
      <td>2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>ITPL Main Road, KIADB Export Promotion Industr...</td>
      <td>the nest - the den bengaluru</td>
      <td>No</td>
      <td>No</td>
      <td>13</td>
      <td>+91 8071117272</td>
      <td>ITPL Main Road, Whitefield</td>
      <td>Bar, Casual Dining</td>
      <td>NaN</td>
      <td>finger food, north indian, continental</td>
      <td>[('Rated 5.0', 'RATED\n  Great ambience , look...</td>
      <td>[]</td>
      <td>Pubs and bars</td>
      <td>Whitefield</td>
      <td>3.4</td>
      <td>1500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>40764 rows × 18 columns</p>
</div>



Normalising all text to lower case


```python
# Apply lowercase transformation to all object columns
for col in recom_df.select_dtypes(include='object').columns:
    recom_df[col] = recom_df[col].str.lower()

```

Cleaning Phone columns



```python

# Step 1: Replace newline characters with commas, only for string values
recom_df['phone'] = recom_df['phone'].apply(lambda x: x.replace('\r\n', ',').replace('\n', ',') if isinstance(x, str) else '')

# Step 2: Split multiple phone numbers into a list
recom_df['phone'] = recom_df['phone'].apply(lambda x: x.split(',') if x else [])

# Step 3: Clean, format each phone number, and remove the '+91' prefix if it exists
recom_df['phone'] = recom_df['phone'].apply(lambda numbers: [num.strip().replace(" ", "").lstrip('+91') for num in numbers if num])

# Step 4: Optionally, convert each list of numbers back to a single string (join with commas)
recom_df['phone'] = recom_df['phone'].apply(lambda numbers: ', '.join(numbers) if numbers else '')

```


```python
recom_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st main road, 2nd stage, banashankari, ...</td>
      <td>jalsa</td>
      <td>yes</td>
      <td>yes</td>
      <td>775</td>
      <td>08042297555, 743772233</td>
      <td>banashankari</td>
      <td>casual dining</td>
      <td>pasta, lunch buffet, masala papad, paneer laja...</td>
      <td>north indian, mughlai, chinese</td>
      <td>[('rated 4.0', 'rated\n  a beautiful place to ...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd floor, 80 feet road, near big bazaar, 6th ...</td>
      <td>spice elephant</td>
      <td>yes</td>
      <td>no</td>
      <td>787</td>
      <td>08041714161</td>
      <td>banashankari</td>
      <td>casual dining</td>
      <td>momos, lunch buffet, chocolate nirvana, thai g...</td>
      <td>chinese, north indian, thai</td>
      <td>[('rated 4.0', 'rated\n  had been here for din...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/sanchurrobangalore?cont...</td>
      <td>1112, next to kims medical college, 17th cross...</td>
      <td>san churro cafe</td>
      <td>yes</td>
      <td>no</td>
      <td>918</td>
      <td>663487993</td>
      <td>banashankari</td>
      <td>cafe, casual dining</td>
      <td>churros, cannelloni, minestrone soup, hot choc...</td>
      <td>cafe, mexican, italian</td>
      <td>[('rated 3.0', "rated\n  ambience is not that ...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>3.8</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st floor, annakuteera, 3rd stage, banashankar...</td>
      <td>addhuri udupi bhojana</td>
      <td>no</td>
      <td>no</td>
      <td>88</td>
      <td>620009302</td>
      <td>banashankari</td>
      <td>quick bites</td>
      <td>masala dosa</td>
      <td>south indian, north indian</td>
      <td>[('rated 4.0', "rated\n  great food and proper...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>3.7</td>
      <td>300</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd floor, lakshmi associates, gandhi baza...</td>
      <td>grand village</td>
      <td>no</td>
      <td>no</td>
      <td>166</td>
      <td>8026612447, 01210005</td>
      <td>basavanagudi</td>
      <td>casual dining</td>
      <td>panipuri, gol gappe</td>
      <td>north indian, rajasthani</td>
      <td>[('rated 4.0', 'rated\n  very good restaurant ...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>3.8</td>
      <td>600</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51709</th>
      <td>https://www.zomato.com/bangalore/the-farm-hous...</td>
      <td>136, sap labs india, kiadb export promotion in...</td>
      <td>the farm house bar n grill</td>
      <td>no</td>
      <td>no</td>
      <td>34</td>
      <td>80121279, 00240646</td>
      <td>whitefield</td>
      <td>casual dining, bar</td>
      <td>NaN</td>
      <td>north indian, continental</td>
      <td>[('rated 4.0', 'rated\n  ambience- big and spa...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>3.7</td>
      <td>800</td>
      <td>2</td>
    </tr>
    <tr>
      <th>51711</th>
      <td>https://www.zomato.com/bangalore/bhagini-2-whi...</td>
      <td>139/c1, next to gr tech park, pattandur agraha...</td>
      <td>bhagini</td>
      <td>no</td>
      <td>no</td>
      <td>81</td>
      <td>08065951222</td>
      <td>whitefield</td>
      <td>casual dining, bar</td>
      <td>biryani, andhra meal</td>
      <td>andhra, south indian, chinese, north indian</td>
      <td>[('rated 4.0', 'rated\n  a fine place to chill...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>2.5</td>
      <td>800</td>
      <td>4</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>four points by sheraton bengaluru, 43/3, white...</td>
      <td>best brews - four points by sheraton bengaluru...</td>
      <td>no</td>
      <td>no</td>
      <td>27</td>
      <td>08040301477</td>
      <td>whitefield</td>
      <td>bar</td>
      <td>NaN</td>
      <td>continental</td>
      <td>[('rated 5.0', "rated\n  food and service are ...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>3.6</td>
      <td>1500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>sheraton grand bengaluru whitefield hotel &amp; co...</td>
      <td>chime - sheraton grand bengaluru whitefield ho...</td>
      <td>no</td>
      <td>yes</td>
      <td>236</td>
      <td>08049652769</td>
      <td>itpl main road, whitefield</td>
      <td>bar</td>
      <td>cocktails, pizza, buttermilk</td>
      <td>finger food</td>
      <td>[('rated 4.0', 'rated\n  nice and friendly pla...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>4.3</td>
      <td>2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>itpl main road, kiadb export promotion industr...</td>
      <td>the nest - the den bengaluru</td>
      <td>no</td>
      <td>no</td>
      <td>13</td>
      <td>8071117272</td>
      <td>itpl main road, whitefield</td>
      <td>bar, casual dining</td>
      <td>NaN</td>
      <td>finger food, north indian, continental</td>
      <td>[('rated 5.0', 'rated\n  great ambience , look...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>3.4</td>
      <td>1500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>40764 rows × 18 columns</p>
</div>




```python
# Function to clean the review text
def clean_review(review_text):
    # Remove the 'rated' part at the beginning and any extra spaces
    review_text = re.sub(r'^rated[^\w]*', '', review_text)

    # Remove newline characters
    review_text = review_text.replace("\n", " ")

    # Fix encoding issues (you can add more replacements for any problematic encodings)
    review_text = review_text.replace('ã\x83', '').replace('ã\x82', '').replace('ã\x82', '')

    # Remove extra spaces
    review_text = ' '.join(review_text.split())

    return review_text

# Function to clean the nested tuple inside the list
def clean_nested_tuples(text):
    try:
        # Use ast.literal_eval to safely evaluate the string into a Python object
        list_of_tuples = ast.literal_eval(text)  # Safely converts string to a list of tuples

        # Clean each tuple (both rating and review text)
        cleaned_list = [(rating, clean_review(review)) for rating, review in list_of_tuples]
        return cleaned_list
    except Exception as e:
        # print(f"Error while parsing or cleaning the text: {e}")
        return []

# Apply the cleaning function to each row in the 'reviews_list' column
recom_df['reviews_list'] = recom_df['reviews_list'].apply(clean_nested_tuples)
```


```python
recom_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>address</th>
      <th>name</th>
      <th>online_order</th>
      <th>book_table</th>
      <th>votes</th>
      <th>phone</th>
      <th>location</th>
      <th>rest_type</th>
      <th>dish_liked</th>
      <th>cuisines</th>
      <th>reviews_list</th>
      <th>menu_item</th>
      <th>listed_in(type)</th>
      <th>listed_in(city)</th>
      <th>rating</th>
      <th>Cost_For_two</th>
      <th>Cuisine_Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.zomato.com/bangalore/jalsa-banasha...</td>
      <td>942, 21st main road, 2nd stage, banashankari, ...</td>
      <td>jalsa</td>
      <td>yes</td>
      <td>yes</td>
      <td>775</td>
      <td>08042297555, 743772233</td>
      <td>banashankari</td>
      <td>casual dining</td>
      <td>pasta, lunch buffet, masala papad, paneer laja...</td>
      <td>north indian, mughlai, chinese</td>
      <td>[(rated 4.0, a beautiful place to dine in.the ...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://www.zomato.com/bangalore/spice-elephan...</td>
      <td>2nd floor, 80 feet road, near big bazaar, 6th ...</td>
      <td>spice elephant</td>
      <td>yes</td>
      <td>no</td>
      <td>787</td>
      <td>08041714161</td>
      <td>banashankari</td>
      <td>casual dining</td>
      <td>momos, lunch buffet, chocolate nirvana, thai g...</td>
      <td>chinese, north indian, thai</td>
      <td>[(rated 4.0, had been here for dinner with fam...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>4.1</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://www.zomato.com/sanchurrobangalore?cont...</td>
      <td>1112, next to kims medical college, 17th cross...</td>
      <td>san churro cafe</td>
      <td>yes</td>
      <td>no</td>
      <td>918</td>
      <td>663487993</td>
      <td>banashankari</td>
      <td>cafe, casual dining</td>
      <td>churros, cannelloni, minestrone soup, hot choc...</td>
      <td>cafe, mexican, italian</td>
      <td>[(rated 3.0, ambience is not that good enough ...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>3.8</td>
      <td>800</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.zomato.com/bangalore/addhuri-udupi...</td>
      <td>1st floor, annakuteera, 3rd stage, banashankar...</td>
      <td>addhuri udupi bhojana</td>
      <td>no</td>
      <td>no</td>
      <td>88</td>
      <td>620009302</td>
      <td>banashankari</td>
      <td>quick bites</td>
      <td>masala dosa</td>
      <td>south indian, north indian</td>
      <td>[(rated 4.0, great food and proper karnataka s...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>3.7</td>
      <td>300</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.zomato.com/bangalore/grand-village...</td>
      <td>10, 3rd floor, lakshmi associates, gandhi baza...</td>
      <td>grand village</td>
      <td>no</td>
      <td>no</td>
      <td>166</td>
      <td>8026612447, 01210005</td>
      <td>basavanagudi</td>
      <td>casual dining</td>
      <td>panipuri, gol gappe</td>
      <td>north indian, rajasthani</td>
      <td>[(rated 4.0, very good restaurant in neighbour...</td>
      <td>[]</td>
      <td>buffet</td>
      <td>banashankari</td>
      <td>3.8</td>
      <td>600</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>51709</th>
      <td>https://www.zomato.com/bangalore/the-farm-hous...</td>
      <td>136, sap labs india, kiadb export promotion in...</td>
      <td>the farm house bar n grill</td>
      <td>no</td>
      <td>no</td>
      <td>34</td>
      <td>80121279, 00240646</td>
      <td>whitefield</td>
      <td>casual dining, bar</td>
      <td>NaN</td>
      <td>north indian, continental</td>
      <td>[(rated 4.0, ambience- big and spacious lawn w...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>3.7</td>
      <td>800</td>
      <td>2</td>
    </tr>
    <tr>
      <th>51711</th>
      <td>https://www.zomato.com/bangalore/bhagini-2-whi...</td>
      <td>139/c1, next to gr tech park, pattandur agraha...</td>
      <td>bhagini</td>
      <td>no</td>
      <td>no</td>
      <td>81</td>
      <td>08065951222</td>
      <td>whitefield</td>
      <td>casual dining, bar</td>
      <td>biryani, andhra meal</td>
      <td>andhra, south indian, chinese, north indian</td>
      <td>[(rated 4.0, a fine place to chill after offic...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>2.5</td>
      <td>800</td>
      <td>4</td>
    </tr>
    <tr>
      <th>51712</th>
      <td>https://www.zomato.com/bangalore/best-brews-fo...</td>
      <td>four points by sheraton bengaluru, 43/3, white...</td>
      <td>best brews - four points by sheraton bengaluru...</td>
      <td>no</td>
      <td>no</td>
      <td>27</td>
      <td>08040301477</td>
      <td>whitefield</td>
      <td>bar</td>
      <td>NaN</td>
      <td>continental</td>
      <td>[(rated 5.0, food and service are incomparably...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>3.6</td>
      <td>1500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51715</th>
      <td>https://www.zomato.com/bangalore/chime-sherato...</td>
      <td>sheraton grand bengaluru whitefield hotel &amp; co...</td>
      <td>chime - sheraton grand bengaluru whitefield ho...</td>
      <td>no</td>
      <td>yes</td>
      <td>236</td>
      <td>08049652769</td>
      <td>itpl main road, whitefield</td>
      <td>bar</td>
      <td>cocktails, pizza, buttermilk</td>
      <td>finger food</td>
      <td>[(rated 4.0, nice and friendly place and staff...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>4.3</td>
      <td>2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>51716</th>
      <td>https://www.zomato.com/bangalore/the-nest-the-...</td>
      <td>itpl main road, kiadb export promotion industr...</td>
      <td>the nest - the den bengaluru</td>
      <td>no</td>
      <td>no</td>
      <td>13</td>
      <td>8071117272</td>
      <td>itpl main road, whitefield</td>
      <td>bar, casual dining</td>
      <td>NaN</td>
      <td>finger food, north indian, continental</td>
      <td>[(rated 5.0, great ambience , looking nice goo...</td>
      <td>[]</td>
      <td>pubs and bars</td>
      <td>whitefield</td>
      <td>3.4</td>
      <td>1500</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>40764 rows × 18 columns</p>
</div>




```python
# path = '/content/drive/MyDrive/Data_Set/cleaned_data.csv'
# recom_df.to_csv(path, index=False)

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
