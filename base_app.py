"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
import pickle

# Data dependencies
import pandas as pd

# Visualization dependecies
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk import ngrams as ngrams, word_tokenize
from collections import Counter

nltk.download('punkt')
from wordcloud import WordCloud
pd.set_option('display.max_colwidth', 100)

st.set_page_config(page_title="SynapseAI Tweet Classifer", page_icon=":cloud:", layout="wide")

bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url('https://images.unsplash.com/photo-1605778336817-121ba9819b96?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1282&q=80');
background-size: cover;
background-position: top left;
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}

[data-testid="stToolbar"] {
right: 2rem;
}

[data-testid="stSidebar"] {
background-image: url('https://images.unsplash.com/photo-1610270197941-925ce9015c40?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1074&q=80');
background-size: center;
background-position: center;
}
</style>
"""
st.markdown(bg_img, unsafe_allow_html=True)

# Vectorizer
news_vectorizer = open("resources/count_vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
df = pd.read_csv("resources/train_preprocessed.csv")

def generate_ngrams_from_column(data, column_name, n):
    tokens = []
    for text in data[column_name]:
        tokens.extend(word_tokenize(text))
    n_grams = ngrams(tokens, n)
    return [' '.join(gram) for gram in n_grams]

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title(":green[SynapseAI] Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Data Info", "Exploratory Data Analysis", "Prediction", "Credits"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("""Welcome to SynapseAI, a cutting-edge AI company revolutionizing industries worldwide.
		Our team of experts leverages advanced algorithms, machine learning, and natural language processing
		to deliver innovative solutions. From personalized virtual assistants to data analytics and automation,
		we empower businesses to thrive in the digital era. Join us on this transformative journey.
		Our ML model for sentiment analysis of climate change-related tweets combines the power of natural
		language processing and machine learning techniques to provide a comprehensive understanding of public
		sentiments. By leveraging this model, we can extract meaningful insights from the vast pool of Twitter
		data, enabling a data-driven approach towards addressing climate change and fostering informed decision-making.""")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(df[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		# Vectorizer
		news_vectorizer = open("resources/count_vectorizer.pkl","rb")
		tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_input("Tweet", value="Enter your text here")

		# Load models from .pkl files
		lr = joblib.load(open(os.path.join("resources/log_reg_model.pkl"),"rb"))
		mnb = joblib.load(open(os.path.join("resources/multinomial_nb_model_1.pkl"),"rb"))
		cnb = joblib.load(open(os.path.join("resources/complement_nb_model_2.pkl"),"rb"))
		#ridge = joblib.load(open(os.path.join("resources/ridge_model.pkl"),"rb"))
		#sc = pickle.load(open(os.path.join("resources/stackingNB_model_1.pkl"),"rb"))
		

		model_list = [lr, mnb, cnb]

		model = st.selectbox('Select Model', options=model_list)

		# Create a dictionary to map prediction labels to human-readable categories
		label_mapping = {'Anti Climate Change': -1, 'Neutral Tweet': 0, 'Pro Climate Change': 1, 'News': 2}

		def get_key_from_value(dictionary, value):
			for key, val in dictionary.items():
				if val == value:
					return key
			return None

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			prediction = model.predict(vect_text)
			# When model has successfully run, will print prediction
			result = get_key_from_value(label_mapping, prediction)
			# more human interpretable.
			st.success("Tweet Categorized as: {}".format(result))

	# Building the EDA page
	if selection == "Exploratory Data Analysis":
		st.info("Exploratory Data Analysis")
		st.subheader("Tweet Distribution")
    	# Group tweets by sentiments
		groups = df.groupby(by='sentiment').count().cleaned_message
		anti = groups[-1]
		neu = groups[0]
		pro = groups[1]
		news = groups[2]
    	# Create a bar chart
		fig = go.Figure()
		fig.add_trace(go.Pie(
    		labels=['Anti', 'Neutral', 'Pro', 'News'],
    		values=[anti, neu, pro, news],
    		marker_colors=['indianred', 'cyan', 'orange', 'pink'],
    		text=[f'ANTI: {anti}', f'NEU: {neu}', f'PRO: {pro}', f'NEWS: {news}']))
		fig.update_layout(title='Frequency of Sentiments',
		    title_x=0.5,
		    height=500,  # Adjust the height as desired
    		width=700,    # Adjust the width as desired
		    title_font=dict(size=24),  # Increase the title font size
    		showlegend=True,
    		legend=dict(font=dict(size=20)),  # Increase the legend font size
    		annotations=[dict(font=dict(size=20))]  # Increase the annotation font size
			)
		# Show the figure
		st.plotly_chart(fig, use_container_width=True)


		st.subheader("Tweets Analysis")
		# Tokenize words
		words = df['cleaned_message'].apply(nltk.word_tokenize)
    	# Flatten the list of words
		all_words = [word for sublist in words for word in sublist]
    	# Calculate frequency distribution
		frequency_dist = nltk.FreqDist(all_words)
    	# Create a DataFrame of the top 25 words
		temp = pd.DataFrame(frequency_dist.most_common(20), columns=['word', 'count'])
    	# Create a bar plot
		fig = px.bar(temp, x='word', y='count', title='Top words')
    	# Rotate the x-ticks vertically
		fig.update_layout(xaxis_tickangle=90,
		    title_font=dict(size=24),  # Increase the title font size
    		annotations=[dict(font=dict(size=20))]  # Increase the annotation font size
		)
    	# Show the plot
		st.plotly_chart(fig, use_container_width=True)

		st.subheader("ngrams analysis")
		selected_ngram = st.selectbox("Select n-gram", ["Bigram", "Trigram"])
		column_name = 'cleaned_message'
		if selected_ngram == "Bigram":
			n = 2
		else:
			n = 3
		ngram_list = generate_ngrams_from_column(df, column_name, n)
		ngram_counter = Counter(ngram_list)
		most_common_ngrams = ngram_counter.most_common(15)  # Get the 15 most common n-grams
		most_common_ngrams.sort(key=lambda x: x[1], reverse=True)  # Sort by frequency in descending order
		labels, values = zip(*most_common_ngrams)
		data = pd.DataFrame({"N-gram": labels, "Frequency": values})
		
		# Create bar chart using Plotly
		fig = go.Figure(data=[go.Bar(x=labels, y=values)])
		fig.update_layout(
            xaxis_title=f"{selected_ngram}s",
            yaxis_title="Frequency",
            title=f"Top 15 Most Common {selected_ngram}s",
	    	title_font=dict(size=24),  # Increase the title font size
    		annotations=[dict(font=dict(size=20))]  # Increase the annotation font size
        )
		st.plotly_chart(fig)

		st.subheader("Wordclouds")
		# Start with one review:
		df_anti = df[df['sentiment']==-1]
		df_neutral = df[df['sentiment']==0]
		df_pro = df[df['sentiment']==1]
		df_news = df[df['sentiment']==2]
		
		# Flatten to lists
		tweet_all = df['cleaned_message'].tolist()
		tweet_anti = df_anti['cleaned_message'].tolist()
		tweet_neutral = df_neutral['cleaned_message'].tolist()
		tweet_pro = df_pro['cleaned_message'].tolist()
		tweet_news = df_news['cleaned_message'].tolist()

		# Join list to a single string
		tweet_all = ' '.join(tweet_all)
		tweet_anti = ' '.join(tweet_anti)
		tweet_pro = ' '.join(tweet_pro)
		tweet_neu = ' '.join(tweet_neutral)
		tweet_news = ' '.join(tweet_news)

		# Exclude the top 5 most occurring words
		excluded_words = set([word for word, _ in frequency_dist.most_common(4)])

		# Split lists to words
		tweet_all = tweet_all.split()
		tweet_anti = tweet_anti.split()
		tweet_neu = tweet_neu.split()
		tweet_news = tweet_news.split()
		tweet_pro = tweet_pro.split()

		# Filter out the excluded words from the original list
		tweet_all = [word for word in tweet_all]
		filtered_news = [word for word in tweet_news if word not in excluded_words]
		filtered_pro = [word for word in tweet_pro if word not in excluded_words]
		filtered_anti = [word for word in tweet_anti if word not in excluded_words]
		filtered_neu = [word for word in tweet_neu if word not in excluded_words]

		# Convert the filtered words back into a single string
		tweet_all = ' '.join(tweet_all)
		filtered_news = ' '.join(filtered_news)
		filtered_pro = ' '.join(filtered_pro)
		filtered_anti = ' '.join(filtered_anti)
		filtered_neu = ' '.join(filtered_neu)

		wordcloud = WordCloud(width=400, height=200, max_font_size=50, max_words=100, background_color="white")

		fig, ax = plt.subplots(5, 1, figsize  = (25, 12), dpi=30)
    	# Create and generate a word cloud image:
		wordcloud_all = WordCloud(width=400, height=300, max_font_size=50, max_words=100, background_color="white").generate(tweet_all)
		wordcloud_anti = WordCloud(width=400, height=300, max_font_size=50, max_words=100, background_color="white").generate(filtered_anti)
		wordcloud_neutral = WordCloud(width=400, height=300, max_font_size=50, max_words=100, background_color="white").generate(filtered_neu)
		wordcloud_pro = WordCloud(width=400, height=300, max_font_size=50, max_words=100, background_color="white").generate(filtered_pro)
		wordcloud_news = WordCloud(width=400, height=300, max_font_size=50, max_words=100, background_color="white").generate(filtered_news)

		# Display the generated image:
		ax[0].imshow(wordcloud_all, interpolation='bilinear')
		ax[0].set_title('All Tweets', fontsize=15)
		ax[0].axis('off')
		ax[1].imshow(wordcloud_news, interpolation='bilinear')
		ax[1].set_title('News Tweets',fontsize=15)
		ax[1].axis('off')
		ax[2].imshow(wordcloud_pro, interpolation='bilinear')
		ax[2].set_title('Pro Tweets',fontsize=15)
		ax[2].axis('off')
		ax[3].imshow(wordcloud_anti, interpolation='bilinear')
		ax[3].set_title('Anti Tweets',fontsize=15)
		ax[3].axis('off')
		ax[4].imshow(wordcloud_neutral, interpolation='bilinear')
		ax[4].set_title('Neutral Tweets',fontsize=15)
		ax[4].axis('off')
		# Show the plot
		st.pyplot(fig)

	# Building the Data Info page
	if selection == "Data Info":
		st.info('Data Information')
		st.subheader("Data Header")
		st.dataframe(df)

		st.subheader('Data Statistics')
		st.write(df.describe())

	# Building the Credits page
	if selection == "Credits":
		st.info('Credits')
		st.subheader('Our Team')
		st.text("""Ajirioghene Oguh\t\tProject Lead\n\nAdeyemo Abdulmalik\t\tTechnical Lead\n\nVirtue-ann Michael\t\tAdmin Lead\n\nAbeeb Adeola Adeshina\t\tMember\n\nMutiso Stephen\t\t\tMember\n\nFolarin Adekemi\t\t\tMember""")
		st.subheader('Images')
		st.text("""Background Image\t\tChristian Lue (unsplash.com)\n\nSidebar Image\t\t\tJohn Cameron (unsplash.com)""")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
