# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:50:27 2018

@author: sriniv11
"""


from nltk.corpus import movie_reviews 
 
# Total reviews
print (len(movie_reviews.fileids())) # Output: 2000
 
# Review categories
print (movie_reviews.categories()) # Output: [u'neg', u'pos']
 
# Total positive reviews
print (len(movie_reviews.fileids('pos'))) # Output: 1000
 
# Total negative reviews
print (len(movie_reviews.fileids('neg'))) # Output: 1000
 
positive_review_file = movie_reviews.fileids('pos')[0] 
print (positive_review_file) # Output: pos/cv000_29590.txt