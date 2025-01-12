import nltk
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
import re
import emoji
from sklearn.feature_selection import SelectKBest, chi2

def preprocess_text(text: str):
    

    # lower casing Turkish Text, Don't use str.lower :)
    text = text.casefold()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove special characters and punctuation
    # HERE THE EMOJIS stuff are being removed, you may want to keep them :D
    text = re.sub(r'[^a-zçğıöşü0-9\s#@]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


corpus = []

# to keep the label order
train_usernames = []

for username, posts in username2posts_train.items():
  train_usernames.append(username)

  # aggregating the posts per user
  cleaned_captions = []
  for post in posts:
    post_caption = post.get("caption", "")
    if post_caption is None:
      continue

    post_caption = preprocess_text(post_caption)

    if post_caption != "":
      cleaned_captions.append(post_caption)


  # joining the posts of each user with a \n
  user_post_captions = "\n".join(cleaned_captions)
  corpus.append(user_post_captions)



custom_stopwords = list(set(turkish_stopwords).union({
    'the'
}))
vectorizer = TfidfVectorizer(stop_words=turkish_stopwords, max_features=15000,min_df=10,sublinear_tf=True,smooth_idf=False,ngram_range=(1, 3))

# fit the vectorizer
vectorizer.fit(corpus)

# transform the data into vectors
x_post_train = vectorizer.transform(corpus)
y_train = [username2_category.get(uname, "NA") for uname in train_usernames]
feature_names = vectorizer.get_feature_names_out()


chi2_selector = SelectKBest(chi2, k=5000) 
x_post_train_selected = chi2_selector.fit_transform(x_post_train, y_train)

selected_feature_indices = chi2_selector.get_support(indices=True)
selected_feature_names = np.array(feature_names)[selected_feature_indices]



df_tfidf = pd.DataFrame(x_post_train_selected.toarray(), columns=selected_feature_names)

print(df_tfidf.sum().sort_values(ascending=False).head(30))


'''
# Inspect the frequency of each word
df_tfidf = pd.DataFrame(x_post_train.toarray(), columns=feature_names)

# Show the most frequent words (words in many posts)
print(df_tfidf.sum().sort_values(ascending=False).head(30))

'''


test_usernames = []
test_corpus = []
for username, posts in username2posts_test.items():
  test_usernames.append(username)
  # aggregating the posts per user
  cleaned_captions = []
  for post in posts:
    post_caption = post.get("caption", "")
    if post_caption is None:
      continue

    post_caption = preprocess_text(post_caption)

    if post_caption != "":
      cleaned_captions.append(post_caption)

  user_post_captions = "\n".join(cleaned_captions)
  test_corpus.append(user_post_captions)


# Just transforming! No Fitting!!!!!
x_post_test = vectorizer.transform(test_corpus)




# Transform the test set using the same feature selection
x_post_test_selected = chi2_selector.transform(x_post_test)

df_tfidf = pd.DataFrame(x_post_train_selected.toarray(), columns=selected_feature_names)
df_tfidf.head(2)