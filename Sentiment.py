import tensorflow as tf
import keras as keras
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


imdb_reviews=pd.read_csv("imdb_reviews.csv")
test_reviews=pd.read_csv("test_reviews.csv")


imdb_reviews.head()

test_reviews.head()


word_index=pd.read_csv("word_indexes.csv")



word_index.head()



word_index=dict(zip(word_index.Words,word_index.Indexes))


def review_encoder(text):
  arr=[word_index[word] for word in text]
  return arr


train_data,train_labels=imdb_reviews['Reviews'],imdb_reviews['Sentiment']
test_data, test_labels=test_reviews['Reviews'],test_reviews['Sentiment']


train_data=train_data.apply(lambda review:review.split())
test_data=test_data.apply(lambda review:review.split())


train_data=train_data.apply(review_encoder)
test_data=test_data.apply(review_encoder)


train_data.head()

def encode_sentiments(x):
  if x=='positive':
    return 1
  else:
    return 0

train_labels=train_labels.apply(encode_sentiments)
test_labels=test_labels.apply(encode_sentiments)


train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=500)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=500)


model=keras.Sequential([keras.layers.Embedding(10000,16,input_length=500),
                        keras.layers.GlobalAveragePooling1D(),
                        keras.layers.Dense(16,activation='relu'),
                        keras.layers.Dense(1,activation='sigmoid')])




model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#training the model
history=model.fit(train_data,train_labels,epochs=30,batch_size=512,validation_data=(test_data,test_labels))


loss,accuracy=model.evaluate(test_data,test_labels)


index=np.random.randint(1,1000)
user_review=test_reviews.loc[index]
print(user_review)


user_review=test_data[index]
user_review=np.array([user_review])
if (model.predict(user_review)>0.5).astype("int32"):
  print("positive sentiment")
else:
  print("negative sentiment")
