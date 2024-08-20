from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from BookRecommendationModel import BookRecommendationModel  # 사용자 정의 모델 클래스가 있는 모듈

app = Flask(__name__)

# 현재 디렉토리에 있는 모델 파일 경로
model_path = os.path.join(os.getcwd(), 'personalRecommend_ver4.keras')

# 모델 로드
model = tf.keras.models.load_model(model_path)

# 추천 함수
def recommend_books_with_model(user_id, model, userBook, num_recommendations=4):
    user_id_encoded = user_id_mapping[user_id]
    category_vector = [0] * num_categories

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
        return {"message": "추천할 책이 없습니다."}  # JSON 형식으로 변경

    recommended_books_info = book[book['book_id'].isin(recommended_books)]
    recommended_books_info = recommended_books_info.head(num_recommendations)

    return recommended_books_info[['book_id', 'title']].to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json.get('user_id')

    # 유효성 검사
    if user_id not in user_id_mapping:
        return jsonify({"error": "Invalid user ID"}), 400

    # 추천 함수 호출
    recommended_books = recommend_books_with_model(user_id, model, userBook, num_recommendations=4)

    # 추천 결과가 메시지인 경우
    if isinstance(recommended_books, dict) and "message" in recommended_books:
        return jsonify(recommended_books), 404

    return jsonify(recommended_books)

if __name__ == '__main__':
    app.run(port=5000)
