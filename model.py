### --- TFRS를 사용하여 추천 모델을 구축 --- ###

import tensorflow as tf
from keras import layers
from keras import Model
# from tensorflow.python.keras.saving import register_keras_serializable

# @register_keras_serializable()
class BookRecommendationModel(tf.keras.Model):
    def __init__(self, num_users, num_books, num_categories, trainable=True, name='book_recommendation_model', **kwargs):
        super(BookRecommendationModel, self).__init__(trainable=trainable, name=name, **kwargs)
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
        return tf.keras.losses.binary_crossentropy(labels, predictions)

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
            'trainable': self._trainable,  # _trainable 대신 trainable을 사용
            'name': self.name  # name 추가
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 'dtype'와 같은 추가 인자를 필터링
        config.pop('dtype', None)
        return cls(
            num_users=config['num_users'],
            num_books=config['num_books'],
            num_categories=config['num_categories'],
            trainable=config.get('trainable', True),
            name=config.get('name', 'book_recommendation_model')  # config에서 직접 가져오기
        )
