Project Description: 

This Programming Tutor is an intelligent assistant designed to help learners with fundamental C programming concepts. Built using Python and machine learning, this application provides instant answers to common C programming questions through a user-friendly GUI interface.


Key Features: 

Question-Answer System - Answers questions about C programming basics including variables, loops, functions, memory allocation, and more.

Machine Learning Backend - Uses a supervised learning approach with a Multinomial Naive Bayes classifier trained on a curated dataset of C programming questions and answers.

Natural Language Processing - Implements text preprocessing (stopword removal, stemming) and TF-IDF vectorization to understand user queries.

Confidence Thresholding - Only provides answers when the model is confident, otherwise directs users to external resources.

User-Friendly Interface - Clean Tkinter GUI with chat-like interaction, clear formatting, and responsive design.


Technical Implementation:

Supervised Learning Algorithm - Multinomial Naive Bayes classifier trained on question-answer pairs.

Text Processing - NLTK for stopword removal and Porter Stemmer for text normalization.

Feature Extraction - TF-IDF with n-grams (1,2) for better context understanding.

Python Libraries - Uses scikit-learn for ML, pandas for data handling, and tkinter for GUI.


Potential Applications:

Learning tool for beginner programmers

Quick reference guide for C programming concepts

Template for building similar Q&A systems in other domains

Demonstration of NLP and ML in educational applications


GitHub Repository: 

View on GitHub https://github.com/nisadi/Tutor/tree/main
