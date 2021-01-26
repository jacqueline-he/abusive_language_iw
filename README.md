# Detecting and Classifying Online Abusive Content via BERT-based Transfer Learning

## Abstract ## 
The meteoric growth of social media networks in the past decade has altered the landscape of
cyberspace communication. The rapid dissemination of abusive language on online platforms has
generated much interest in automated systems that can detect and classify such content. However,
one recent area of concern is that many widely-used hate speech corpora suggest evidence
of systematic racial bias; therefore, classifiers trained on these datasets would unintentionally
generate discriminatory results. In this work, we first leveraged transfer learning to construct a
trained classifier that can correctly differentiate between normal, abusive, spam, and hate speech
content on Twitter, using a powerful transformer-based model called BERT (Bidirectional Encoder
Representations from Transformers). With an overall accuracy of 81.1% and an F1 score of 0.819,
we achieved competitive performance relative to baseline classifiers built from other deep learning
architectures. Next, to mitigate in-dataset racial bias, we applied a regularization mechanism that
reweights bigrams in training data, thereby minimizing the influence of highly correlated phrases.
Experimental results indicate that this approach substantially reduces the presence of racial bias,
while only minimally affecting classifier performance.

## Paper ##
Please email me at [jyh@princeton.edu] for the full draft!

