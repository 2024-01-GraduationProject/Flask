### --- TFRS를 사용하여 추천 모델을 구축 --- ###

import tensorflow as tf
from keras import layers
from keras import Model
from keras.saving import register_keras_serializable

@register_keras_serializable()
class BookRecommendationModel(tf.keras.Model):
    def __init__(self, num_users, num_books, num_categories, trainable=True, name='book_recommendation_model'):
        super(BookRecommendationModel, self).__init__(trainable=trainable, name=name)
        self.user_embedding = tf.keras.layers.Embedding(input_dim=num_users + 1, output_dim=128)
        self.book_embedding = tf.keras.layers.Embedding(input_dim=num_books + 1, output_dim=128)
        self.category_embedding = tf.keras.layers.Embedding(input_dim=num_categories + 1, output_dim=64)
        self.dense_layer = tf.keras.layers.Dense(96, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_categories, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        user_ids = inputs[0]
        category_ids = inputs[1]

        user_embedded = self.user_embedding(user_ids)  # 사용자 임베딩
        category_processed = self.category_embedding(category_ids)  # 카테고리 임베딩
        category_processed = tf.reduce_mean(category_processed, axis=1)  # 카테고리 임베딩의 평균

        combined = tf.concat([user_embedded, category_processed], axis=-1)  # 임베딩 결합

        x = self.dense_layer(combined)  # Dense 레이어 통과

        return self.output_layer(x)  # 최종 출력

    def compute_loss(self, predictions, labels):
        return tf.keras.losses.binary_crossentropy(labels, predictions)  # 필요에 따라 categorical_crossentropy로 변경 가능

    def build(self, input_shape):
        self.user_embedding.build((None,))
        self.book_embedding.build((None,))
        self.category_embedding.build((None, input_shape[1][-1]))  # 카테고리 임베딩
        self.dense_layer.build((None, self.user_embedding.output_dim + self.category_embedding.output_dim))
        self.output_layer.build((None, self.dense_layer.units))

    def get_config(self):
        config = super(BookRecommendationModel, self).get_config()
        config.update({
            'num_users': self.user_embedding.input_dim - 1,
            'num_books': self.book_embedding.input_dim - 1,
            'num_categories': self.category_embedding.input_dim - 1,
            'trainable': self._trainable,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)  # 필요한 인자만 전달

# # 모델 인스턴스 생성
# num_users = len(user_id_mapping)  # 사용자 수
# num_books = len(book_id_mapping)    # 책 수
# num_categories = len(category_to_index)  # 카테고리 수

# model = BookRecommendationModel(num_users, num_books, num_categories)

# # 모델 빌드 및 요약
# model.build(input_shape=[(None,), (None, len(category_to_index))])  # user_ids, category_ids
# model.summary()
