from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from keras import layers
from keras import Model
from keras.saving import register_keras_serializable

# Flask 애플리케이션 생성
app = Flask(__name__)

@register_keras_serializable()
class BookRecommendationModel(tf.keras.Model):
    def __init__(self, num_users, num_books, num_categories, trainable=True):
        super(BookRecommendationModel, self).__init__(trainable=trainable)
        self.user_embedding = tf.keras.layers.Embedding(input_dim=num_users + 1, output_dim=128)
        self.book_embedding = tf.keras.layers.Embedding(input_dim=num_books + 1, output_dim=128)
        self.category_embedding = tf.keras.layers.Embedding(input_dim=num_categories + 1, output_dim=64)
        self.dense_layer = tf.keras.layers.Dense(96, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_categories, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        user_ids = inputs[0]
        category_ids = inputs[1]

        user_embedded = self.user_embedding(user_ids)
        category_processed = self.category_embedding(category_ids)
        category_processed = tf.reduce_mean(category_processed, axis=1)

        combined = tf.concat([user_embedded, category_processed], axis=-1)
        x = self.dense_layer(combined)

        return self.output_layer(x)

    def compute_loss(self, predictions, labels):
        return tf.keras.losses.binary_crossentropy(labels, predictions)

    def build(self, input_shape):
        self.user_embedding.build((None,))
        self.book_embedding.build((None,))
        self.category_embedding.build((None, input_shape[1][-1]))
        self.dense_layer.build((None, self.user_embedding.output_dim + self.category_embedding.output_dim))
        self.output_layer.build((None, self.dense_layer.units))

# 사용자 수, 책 수, 카테고리 수 설정
num_users = (1000)  # 예시값
num_books = (30)    # 예시값
num_categories = (10)  # 예시값

# 모델 인스턴스 생성
model = BookRecommendationModel(num_users, num_books, num_categories)

# 모델 빌드 및 요약
model.build(input_shape=[(None,), (None, 10)])
model.summary()

# 추천 함수
def recommend_books_with_model(user_id, model, userBook, num_recommendations=4):
    user_id_encoded = user_id_mapping[user_id]
    category_vector = [0] * 10

    user_categories = final_result[final_result['user_id'] == user_id]['category_id'].values
    if len(user_categories) > 0:
        for categories in user_categories:
            for i, val in enumerate(categories):
                if val == 1:
                    category_vector[i] = 1

    user_ids_input = tf.constant([user_id_encoded])
    category_ids_input = tf.constant([category_vector])

    predictions = model([user_ids_input, category_ids_input])

    recommended_book_indices = tf.argsort(predictions, direction='DESCENDING')[:num_recommendations]
    recommended_book_indices = recommended_book_indices.numpy().flatten()
    recommended_books = [list(book_id_mapping.keys())[index] for index in recommended_book_indices]

    read_books = userBook[userBook['user_id'] == user_id]['book_id'].values.tolist()
    recommended_books = [book for book in recommended_books if book not in read_books]

    if len(recommended_books) >= num_recommendations:
        recommended_books = np.random.choice(recommended_books, size=num_recommendations, replace=False).tolist()
    elif len(recommended_books) == 0:
        return "추천할 책이 없습니다."

    recommended_books_info = book[book['book_id'].isin(recommended_books)]
    recommended_books_info = recommended_books_info.head(num_recommendations)

    return recommended_books_info[['book_id', 'title']].to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']

    # 추천 함수 호출
    recommended_books = recommend_books_with_model(user_id, model, userBook, num_recommendations=4)

    return jsonify(recommended_books)

if __name__ == '__main__':
    app.run(port=5000)



