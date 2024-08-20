import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from models import BookRecommendationModel

# 데이터셋 구성
# final_result는 이미 정의된 데이터프레임이라고 가정
user_ids = tf.data.Dataset.from_tensor_slices(final_result['user_id'].values)
max_length = max(final_result['category_id'].apply(len))
padded_category = pad_sequences(final_result['category_id'].tolist(), maxlen=max_length, padding='post', value=0)
padded_category_dataset = tf.data.Dataset.from_tensor_slices(padded_category)

# 데이터셋 결합
user_book_interactions = tf.data.Dataset.zip((user_ids, padded_category_dataset)).shuffle(buffer_size=30)

# 검증 데이터셋과 훈련 데이터셋 나누기
val_dataset = user_book_interactions.take(5)
train_dataset = user_book_interactions.skip(5)

# 배치 크기 설정
train_dataset = train_dataset.batch(16)
val_dataset = val_dataset.batch(5)

# 모델 인스턴스 생성
num_users = len(user_id_mapping)  # 사용자 수
num_books = len(book_id_mapping)    # 책 수
num_categories = len(category_to_index)  # 카테고리 수
model = BookRecommendationModel(num_users, num_books, num_categories)

# 옵티마이저 생성
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 훈련 과정
epochs = 10
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # 훈련 과정
    for step, (user_ids, combined_categories) in enumerate(train_dataset):
        user_ids = tf.cast(user_ids, dtype=tf.int32)

        with tf.GradientTape() as tape:
            predictions = model((user_ids, combined_categories))
            loss = model.compute_loss(predictions, combined_categories)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if step % 100 == 0:
            print(f"Step {step}, Train Loss: {loss.numpy()}")

    train_loss_history.append(loss.numpy())

    # 검증 과정
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for val_user_ids, val_combined_categories in val_dataset:
        val_user_ids = tf.cast(val_user_ids, dtype=tf.int32)
        val_predictions = model((val_user_ids, val_combined_categories))
        val_loss += model.compute_loss(val_predictions, val_combined_categories)

        predicted_classes = tf.round(tf.nn.sigmoid(val_predictions))
        correct_predictions += tf.reduce_sum(tf.cast(tf.equal(predicted_classes, tf.cast(val_combined_categories, tf.float32)), tf.float32))
        total_predictions += tf.size(val_combined_categories)

    val_loss /= len(val_dataset)
    val_loss_history.append(val_loss.numpy())

    accuracy = correct_predictions / tf.cast(total_predictions, tf.float32)
    print(f"Validation Loss: {val_loss.numpy()}, Validation Accuracy: {accuracy.numpy()}")

# 모델 저장
model.save('personalRecommend_ver4.keras')
