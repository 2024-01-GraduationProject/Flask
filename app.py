from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import requests  # 추가된 부분
from keras.models import load_model
from BookRecommendationModel import BookRecommendationModel  # 사용자 정의 모델 클래스가 있는 모듈

app = Flask(__name__)

# 현재 디렉토리에 있는 모델 파일 경로
model_path = os.path.join(os.getcwd(), 'personalRecommend_ver4.keras')

# 모델 로드
model = tf.keras.models.load_model(model_path)

# 추천 함수
def recommend_books_with_model(user_id, model, userBook, num_recommendations=4):
    # 사용자 ID 인코딩
    user_id_encoded = user_id_mapping.get(user_id)
    if user_id_encoded is None:
        return {"error": "User ID not found"}, 400
    
    # 카테고리 벡터 초기화
    category_vector = [0] * num_categories

    # 사용자의 카테고리 가져오기
    user_categories = final_result[final_result['user_id'] == user_id]['category_id'].values
    if len(user_categories) > 0:
        for categories in user_categories:
            for i, val in enumerate(categories):
                if val == 1:
                    category_vector[i] = 1

    # 모델 입력 데이터 준비
    user_ids_input = tf.constant([user_id_encoded])
    category_ids_input = tf.constant([category_vector])

    # 모델 예측
    predictions = model([user_ids_input, category_ids_input])

    # 추천할 책 인덱스 정렬
    recommended_book_indices = tf.argsort(predictions, direction='DESCENDING')[:num_recommendations]
    recommended_book_indices = recommended_book_indices.numpy().flatten()
    recommended_books = [list(book_id_mapping.keys())[index] for index in recommended_book_indices]

    # 이미 읽은 책 제외
    read_books = userBook[userBook['user_id'] == user_id]['book_id'].values.tolist()
    recommended_books = [book for book in recommended_books if book not in read_books]

    # 추천 책 개수에 따른 처리
    if len(recommended_books) >= num_recommendations:
        recommended_books = np.random.choice(recommended_books, size=num_recommendations, replace=False).tolist()
    elif len(recommended_books) == 0:
        return {"message": "추천할 책이 없습니다."}, 404  # JSON 형식으로 변경

    # 추천된 책의 정보 가져오기
    recommended_books_info = book[book['book_id'].isin(recommended_books)]
    recommended_books_info = recommended_books_info.head(num_recommendations)

    return recommended_books_info[['book_id', 'title']].to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get('user_id')

    # 유효성 검사
    if user_id not in user_id_mapping:
        return jsonify({"error": "Invalid user ID"}), 400

    # 스프링 부트에서 전처리된 데이터 가져오기
    try:
        response = requests.get('http://localhost:8080/preprocess')  # 스프링 부트 전처리 엔드포인트
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch preprocessing data"}), 500

        preprocess_data = response.json()  # JSON 데이터로 변환
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 추천 함수 호출
    recommended_books = recommend_books_with_model(user_id, model, userBook, num_recommendations=4)

    # 추천 결과가 메시지인 경우
    if isinstance(recommended_books, dict) and "message" in recommended_books:
        return jsonify(recommended_books), 404

    return jsonify(recommended_books)

if __name__ == '__main__':
    app.run(port=5000)

