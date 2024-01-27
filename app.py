import subprocess
import sys


# Upgrade pip
subprocess.call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

# Install necessary dependencies
subprocess.call([sys.executable, "-m", "pip", "install", "nltk"])
import altair as alt
import streamlit as st
import pandas as pd
import nltk

from ngram import NGRAM, SmoothNGRAM, words, test_words

st.header("N-GRAM Model:  A Statistical Language Model")
st.write(
    "This n-gram model was trained on austen-emma.txt of the gutenberg corpus and was tested on austen-persuasion.txt")

num_top_trigrams = st.number_input("Enter the number of top trigrams to display:", min_value=1, value=10)
st.subheader(f"The frequencies of the top {num_top_trigrams} trigrams are: ")
ngram_model = NGRAM(3, words)
top_trigrams = ngram_model.n_gram_counter.most_common(num_top_trigrams)
df_top_trigrams = pd.DataFrame(top_trigrams, columns=["Trigram", "Frequency"])
st.table(df_top_trigrams)

num_top_bigrams = st.number_input("Enter the number of top bigrams to display:", min_value=1, value=10)
st.subheader(f"The frequencies of the top {num_top_bigrams} bigrams are: ")
top_bigrams = ngram_model.n_minus_1_gram_counter.most_common(num_top_bigrams)
df_top_bigrams = pd.DataFrame(top_bigrams, columns=["Bigram", "Frequency"])
st.table(df_top_bigrams)

st.subheader("Calculate the probability of a word following a given n-1 gram: ")
selected_model = st.selectbox("Would you like to apply Laplace Smoothing:", ["No", "Yes"], key="prob_select")
n = st.number_input("Enter 'n':", min_value=2, value=3)
sentence = st.text_input(
    "Enter the n-1 gram sentence prefix. If you enter sentence greater than n-1, only the last n-1 tokens will be considered",
    "i do")
word = st.text_input(
    "Enter the word whose probability you would like to check given n-1 gram sentence. Note: if you enter more tha one word, only last word would be considered",
    "not")
n_minus_1_gram = tuple(nltk.word_tokenize(sentence.lower())[-n + 1:])
word = nltk.word_tokenize(word.lower())[-1]
if selected_model == "Yes":
    ngram_model = SmoothNGRAM(n, words)
else:
    ngram_model = NGRAM(n, words)
st.write(f"The probability of '{word}' given {n_minus_1_gram} is :",
         ngram_model.calculate_probability(word, n_minus_1_gram))

st.subheader("Predict the next word given a prefix sentence")
selected_model = st.selectbox("Would you like to apply Laplace Smoothing:", ["No", "Yes"], key="nextword_select")
n = st.number_input("Enter 'n':", min_value=2, value=3, key="n_input")
sentence = st.text_input("Enter the sequence of words", "i do")
if selected_model == "Yes":
    ngram_model = SmoothNGRAM(n, words)
else:
    ngram_model = NGRAM(n, words)
st.write(f"The next word predicted by the ngram model for '{sentence.lower()}' is: ",
         f"**{ngram_model.predict_next_word(sentence.lower())}**")

st.subheader("Generate a sentence of fixed length")
selected_model = st.selectbox("Would you like to apply Laplace Smoothing:", ["No", "Yes"], key="generate_select")
n = st.number_input("Enter 'n':", min_value=2, value=3, key="n_input_generate")
l = st.number_input("Enter the length of the sentence to be generated:", min_value=10, value=20)
prefix = st.text_input("Enter the prefix: ", "emma by")
if selected_model == "Yes":
    ngram_model = SmoothNGRAM(n, words)
else:
    ngram_model = NGRAM(n, words)
st.write("Generated sentence is:", f"**{ngram_model.generate_sentence(l, prefix.lower())}**")

st.subheader("Perplexity comparision of different models")
st.write(
    "The ngram models were evaluated on perplexity scores. For this evaluation the model was trained on austen-emma.txt of the the gutenberg corpus and was tested on austen-persuasion.txt")
st.write(
    "Only Laplace Smoothed ngram models are evaluated for perplexity, since the unsmoothed model throws division by zero exception, as most of the sequences in the testset were not seen in the train set")
n = []
perplexities = []
for i in range(2, 10):
    ngram_model = SmoothNGRAM(n=i, words=words)
    perplexity = ngram_model.perplexity(test_words)
    n.append(i)
    perplexities.append(perplexity)

data = {'n': n, 'Perplexity Score': perplexities}
df = pd.DataFrame(data)
st.table(df)
chart = alt.Chart(df).mark_line().encode(
    x='n',
    y='Perplexity Score',
).properties(
    width=500,  # Adjust the width as needed
    height=300,  # Adjust the height as needed
)

# Customize axis and title font sizes
chart = chart.configure_axis(
    labelFontSize=12,
    titleFontSize=14,
).configure_title(
    fontSize=16
)

# Display the Altair chart using st.altair_chart
st.altair_chart(chart, use_container_width=True)

st.subheader("Testing with different n-1 gram (bigram) inputs to see how it predicts the next word:")


def generate_predictions_table(model, top_bigrams):
    predictions = []

    for bigram in top_bigrams:
        context = ' '.join(bigram[0])
        next_word = model.predict_next_word(context)
        predictions.append({'Context': context, 'Predicted Word': next_word})

    return pd.DataFrame(predictions)


st.text("For unsmoothed model:")
ngram_model = NGRAM(3, words)
unsmoothed_predictions = generate_predictions_table(ngram_model, top_bigrams)
st.table(unsmoothed_predictions)

# For smoothed model
st.text("For smoothed model:")
smoothed_model = SmoothNGRAM(3, words)
smoothed_predictions = generate_predictions_table(smoothed_model, top_bigrams)
st.table(smoothed_predictions)

st.header("Report")
st.write("N-gram models represent a category of statistical language models employed in natural language processing and machine learning. The 'N' in 'N-gram' denotes the quantity of words treated as a single unit. These models operate on the concept that the likelihood of a word in a sequence is contingent upon the preceding N-1 words. 'N' serves as a hyperparameter, and in this project, I conducted experiments with 'n' ranging from 2 to 10, assessing their respective perplexity scores.")
st.write("It is noteworthy that I could calculate perplexity scores exclusively for Laplace-smoothed N-gram models. The computation of perplexity scores for non-smoothed models proved unfeasible, primarily due to recurring division by zero exceptions. This issue arose as I evaluated perplexity using the test data 'austen-persuasion.txt,' while the N-gram model derived its probabilities from the training data 'austen-emma.txt' of the Gutenberg corpus. The majority of N-1 grams in the test set did not occur in the training set, leading to the division by zero error during probability calculations for perplexity.")
st.write("Upon calculating perplexity scores for the smoothed version across various 'n' values of N-gram models, it emerged that an increase in 'n' corresponded to an elevation in perplexity scores. The evaluation spanned from n=2 to 9, with n=2 yielding the lowest perplexity score of 167, while n=9 resulted in a score of 467.")
st.write("Laplace smoothing adversely impacted the model's performance. Upon qualitative analysis of the generated text from both the unsmoothed and smoothed models, the unsmoothed model's text appeared more meaningful and coherent compared to the Laplace-smoothed version.")
