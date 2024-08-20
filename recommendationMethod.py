### --- 훈련된 모델을 사용하여 추천 결과 출력 --- ###

import numpy as np

def recommend_books_with_model(user_id, model, userBook, num_recommendations=4):
    # 사용자 ID를 정수형으로 변환
    user_id_encoded = user_id_mapping[user_id]

    # 카테고리 벡터 초기화
    category_vector = [0] * len(category_to_index)

    # 사용자의 카테고리 정보 가져오기
    user_categories = final_result[final_result['user_id'] == user_id]['category_id'].values
    if len(user_categories) > 0:
        # 원-핫 인코딩된 카테고리 벡터 업데이트
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
    recommended_book_indices = recommended_book_indices.numpy().flatten()  # 1차원 배열로 변환
    recommended_books = [list(book_id_mapping.keys())[index] for index in recommended_book_indices]

    # 사용자가 읽은 책 목록 가져오기
    read_books = userBook[userBook['user_id'] == user_id]['book_id'].values.tolist()
    print(f"사용자가 읽은 책 ID: {read_books}")

    # 추천된 책에서 읽은 책 제외
    recommended_books = [book for book in recommended_books if book not in read_books]
    print(f"추천된 책 (읽은 책 제외): {recommended_books}")

    # 추천된 책이 4개 이상인 경우 랜덤으로 4개 선택
    if len(recommended_books) >= num_recommendations:
        recommended_books = np.random.choice(recommended_books, size=num_recommendations, replace=False).tolist()
    elif len(recommended_books) == 0:
        return "추천할 책이 없습니다."

    # 추천된 책 정보 조회
    recommended_books_info = book[book['book_id'].isin(recommended_books)]

    # 추천된 책 정보를 최대 num_recommendations 개수로 제한
    recommended_books_info = recommended_books_info.head(num_recommendations)

    return recommended_books_info[['book_id', 'title']]

# 예시: 사용자 ID가 user_id인 경우 추천
user_id = 15
recommended_books = recommend_books_with_model(user_id, model, userBook, num_recommendations=4)
print(f"user_id {user_id}인 사용자에게 추천하는 책:")
print(recommended_books)
