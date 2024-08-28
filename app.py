from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from model import BookRecommendationModel # 모델 정의가 있는 파일 import

app = Flask(__name__)
CORS(app) # 모든 경로에 대해 CORS 허용

@app.route("/recommend", methods=['POST']) # 추천을 위한 엔드포인트 정의
def recommend():
    
    ## -- 데이터 전처리 -- ##
    
    # 스프링 부트에서 받은 데이터
    data = request.get_json()
    
    user_id = data.get('userId')
    final_result = data.get('data')
    userBook = data.get('userBook')
    book = data.get('book')
    
    # DataFrame으로 변환
    final_result = pd.DataFrame(final_result)
    userBook = pd.DataFrame(userBook)
    book = pd.DataFrame(book)
    
    # 원-핫 인코딩
    all_categories = set(final_result['combined_category_id'].sum())
    category_to_index = {category: index for index, category in enumerate(all_categories)}

    def one_hot_encode(categories):
        one_hot = [0] * len(category_to_index)
        for category in categories:
            if category in category_to_index:
                one_hot[category_to_index[category]] = 1
        return one_hot

    final_result['category_id'] = final_result['combined_category_id'].apply(one_hot_encode)
    
    # 사용자 및 책 ID를 정수형으로 변환하는 매핑 생성 (1부터 시작)
    user_id_mapping = {user_id: idx + 1 for idx, user_id in enumerate(final_result['user_id'].unique())}
    book_id_mapping = {book_id: idx for idx, book_id in enumerate(userBook['bookId'].unique())}
    
    # 변환된 데이터셋 생성
    final_result['user_id'] = final_result['user_id'].map(user_id_mapping)
    
    # TensorFlow 데이터셋 변환
    user_ids = tf.data.Dataset.from_tensor_slices(final_result['user_id'].values)
    category_ids = tf.data.Dataset.from_tensor_slices(list(final_result['category_id']))
    
    # # 데이터셋 구성
    # user_book_interactions = tf.data.Dataset.zip((user_ids, category_ids)).batch(64)
    
    # 모델 인스턴스 생성
    num_users = len(user_id_mapping)  # 사용자 수
    num_books = len(book_id_mapping)    # 책 수
    num_categories = len(category_to_index)  # 카테고리 수
    
    model = BookRecommendationModel(num_users, num_books, num_categories)
    
    ## -- 훈련된 모델 로드 -- ##
    model.load_weights('personalRecommend_ver5.keras')
    
    ## -- 추천 함수 정의 -- ##
    def recommend_books_with_model(user_id, model, userBook, final_result, num_recommendations=4):
        # 사용자 ID를 정수형으로 변환
        user_id_encoded = user_id_mapping[user_id]

        # 카테고리 벡터 초기화
        category_vector = [0] * len(category_to_index)

        # 사용자의 카테고리 정보 가져오기
        user_categories = final_result[final_result['user_id'] == user_id]['category_id'].values
        if len(user_categories) > 0:
            for categories in user_categories:
                for i, val in enumerate(categories):
                    if val == 1:
                        category_vector[i] = 1

        # 모델에 입력할 데이터 준비
        user_ids_input = tf.constant([user_id_encoded])
        category_ids_input = tf.constant([category_vector])

        # 모델 예측
        predictions = model([user_ids_input, category_ids_input])

        # 예측값을 기반으로 추천할 책 선정
        recommended_book_indices = tf.argsort(predictions, direction='DESCENDING')[:num_recommendations]

        # 추천된 책 ID를 가져오기
        recommended_book_indices = recommended_book_indices.numpy().flatten()
        recommended_books = [list(book_id_mapping.keys())[index] for index in recommended_book_indices]

        # 사용자가 읽은 책 목록 가져오기
        read_books = userBook[userBook['userId'] == user_id]['bookId'].values.tolist()
        
        # 추천된 책에서 읽은 책 제외
        recommended_books = [book for book in recommended_books if book not in read_books]

        # 추천된 책이 4개 이상인 경우 랜덤으로 4개 선택
        if len(recommended_books) >= num_recommendations:
            recommended_books = np.random.choice(recommended_books, size=num_recommendations, replace=False).tolist()
        elif len(recommended_books) == 0:
            return "추천할 책이 없습니다."

        # 추천된 책 정보 조회
        recommended_books_info = book[book['bookId'].isin(recommended_books)]
        recommended_books_info = recommended_books_info.head(num_recommendations)

        return recommended_books_info[['bookId', 'title']].to_dict(orient='records')
    
    ## -- 추천 결과 생성 -- ##
    recommend_books = recommend_books_with_model(user_id, model, userBook, final_result)
    
    print(recommend_books)
    return jsonify(recommend_books)

if __name__ == '__main__':
    app.run(port=5000)
