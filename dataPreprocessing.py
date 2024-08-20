### --- 데이터 전처리 --- ###

import pandas as pd
import tensorflow as tf

# 1단계: userBook과 bookCategoryConnection 병합하여 category_id 찾기
merged_df = userBook.merge(bookCategoryConnection, on='book_id', how='left')

# 2단계: user_id와 category_id로 정렬
sorted_categories = merged_df[['user_id', 'category_id']].dropna().sort_values(by='user_id')

# 3단계: bookTaste와 병합
final_df = sorted_categories.merge(bookTaste[['user_id', 'category_id']], on='user_id', how='outer', suffixes=('_from_book', '_from_taste'))

# 4단계: 필요한 열만 남기고 결합
final_df['combined_category_id'] = final_df.apply(lambda x: list(set([x['category_id_from_book'], x['category_id_from_taste']]) - {None}), axis=1)
final_result = final_df[['user_id', 'combined_category_id']].copy()

# 결측치 처리
final_result['combined_category_id'] = final_result['combined_category_id'].apply(lambda x: [cat for cat in x if pd.notna(cat)])
final_result = final_result.groupby('user_id')['combined_category_id'].agg(lambda x: list(set(sum(x.tolist(), [])))).reset_index()

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

# 최종 결과 출력
print(final_result.head())

# 사용자 및 책 ID를 정수형으로 변환하는 매핑 생성 (1부터 시작)
user_id_mapping = {user_id: idx + 1 for idx, user_id in enumerate(final_result['user_id'].unique())}
book_id_mapping = {book_id: idx for idx, book_id in enumerate(userBook['book_id'].unique())}

# 변환된 데이터셋 생성
final_result['user_id'] = final_result['user_id'].map(user_id_mapping)

# TensorFlow 데이터셋 변환
user_ids = tf.data.Dataset.from_tensor_slices(final_result['user_id'].values)
category_ids = tf.data.Dataset.from_tensor_slices(list(final_result['category_id']))

# 데이터셋 구성
user_book_interactions = tf.data.Dataset.zip((user_ids, category_ids)).batch(64)