# fashion-support-chatbot
The Fashion Support Chatbot is an NLP-based conversational system built using Streamlit that answers customer queries by retrieving and summarizing insights from real e-commerce product reviews.

Instead of using a generative LLM, the chatbot relies on review similarity and metadata filtering to provide factual, explainable responses.

The system helps users ask questions such as:


Is a product true to size?

What are common complaints?

Which dresses are top rated?

Problem Statement


Online shoppers often struggle to interpret large volumes of textual reviews.

This project addresses that problem by:

Structuring raw customer reviews

Computing semantic similarity

Returning concise, review-grounded answers Approach & Architecture

Review text is cleaned and vectorized using TF-IDF

User queries are transformed into the same vector space

Cosine similarity is used to retrieve the most relevant reviews

Metadata such as ratings, department, and recommendation status are aggregated

Results are presented via an interactive Streamlit UI


Technologies Used

Python

Streamlit 

Pandas

Scikit-learn

TF-IDF Vectorization

Cosine Similarity

How to run locally 

live demo

https://fashion-support-chatbot-8c7fcmepq2etnrreljyumv.streamlit.app/

Run the following commands in your terminal:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Future Improvements

Upgrade from TF-IDF to sentence embeddings
Add body-type and fit inference
Deploy with a vector database (FAISS)
Integrate LLM-based summarization

Author

Akshitha Veeranki
