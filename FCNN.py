import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout 
from keras.regularizers import l2 
from keras.layers import Dense, Dropout 
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split

X = pd.read_csv('data/new_train_big_2.csv', index_col='client_id')
gender_train = pd.read_csv('data/train.csv', index_col='client_id') # Таргет по трейну
y = X.join(gender_train, how='inner')['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

num_features = 310  # Количество признаков 
model = Sequential() 
 
# Входной слой 
model.add(Dense(2048, input_dim=num_features, activation='relu')) 
 
# Скрытые слои 
model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.001))) 
model.add(Dropout(0.5)) 
model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.001))) 
model.add(Dropout(0.5)) 
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001))) 
model.add(Dropout(0.5)) 
 
# Выходной слой 
model.add(Dense(1, activation='sigmoid')) 
 
# Компиляция модели 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC']) 
 
model.compile(optimizer=Adam(lr=0.001), 
              loss='binary_crossentropy', 
              metrics=['AUC']) 
 
early_stopping = EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True) 
 
history = model.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_data=(X_test, y_test), 
                    callbacks=[early_stopping])
