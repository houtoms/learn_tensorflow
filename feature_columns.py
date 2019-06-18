import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_numeric():
    # 1. Input features
    price = {'price': [[1.], [2.], [3.], [4.]]}
    builder = _LazyBuilder(price)

    def transform_fn(x):
        return x + 2

    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)
    price_transformed_tensor = price_column._get_dense_tensor(builder)
    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))

    # 2. Feature columns (Dense)
    price_column = feature_column.numeric_column('price',
                                                 normalizer_fn=transform_fn)
    # 3. Feature tensor 
    price_transformed_tensor = feature_column.input_layer(price, [price_column])
    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))

#test_numeric()

def test_bucketized_column():
    # 1. Input features
    price = {'price': [[15.], [5.], [35.], [25.]]} 
    # 2. Feature columns (Dense)
    price_column = feature_column.numeric_column('price')
    # 2. Feature columns (Dense): bucketized_column is both Dense and
    # Categorical
    bucket_price = feature_column.bucketized_column(price_column, [10, 20, 30])
    # 3. Feature tensor 
    price_bucket_tensor = feature_column.input_layer(price, [bucket_price])
    with tf.Session() as session:
        print(session.run([price_bucket_tensor]))

test_bucketized_column()

def test_categorical_identity_column():
    # 1. Input features
    price = {'price': [[3], [1], [2], [0]]} 
    # 2. Feature columns (Sparse)
    identity_feature_column = feature_column.categorical_column_with_identity(
        key='price', num_buckets=4)
    # 2. Feature columns (Dense)
    # Convert the Categorical Column to Dense Column
    indicator_column = feature_column.indicator_column(identity_feature_column)
    # 3. Feature tensor 
    identity_feature_tensor = feature_column.input_layer(
        price, [indicator_column])
    with tf.Session() as session:
        print(session.run([identity_feature_tensor]))

#test_categorical_identity_column()

def test_categorical_column_with_vocabulary_list():
    # 1. Input features
    color_data = {'color': [['R', 'R'], ['G', 'R'], ['B', 'G'], ['A', 'A']]}
    builder = _LazyBuilder(color_data)
    # 2. Feature columns (Sparse)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 2. Feature columns (Dense)
    # Convert the Categorical Column to Dense Column
    color_column_identity = feature_column.indicator_column(color_column)
    # 3. Feature tensor 
    color_dense_tensor = feature_column.input_layer(color_data,
                                                    [color_column_identity])
    with tf.Session() as session:
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

#test_categorical_column_with_vocabulary_list()

def test_categorical_column_with_hash_bucket():
    # 1. Input features
    color_data = {'color': [[2], [5], [-1], [0]]}
    builder = _LazyBuilder(color_data)
    # 2. Feature columns (Sparse)
    color_column = feature_column.categorical_column_with_hash_bucket(
        'color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 2. Feature columns (Dense)
    # Convert the Categorical Column to Dense Column
    color_column_identity = feature_column.indicator_column(color_column)
    # 3. Feature tensor 
    color_dense_tensor = feature_column.input_layer(color_data,
                                                    [color_column_identity])

    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([color_dense_tensor]))

#test_categorical_column_with_hash_bucket()

def test_crossed_column():
    # 1. Input features
    featrues = {
        'price': [['A'], ['B'], ['C'], ['C']],
        'color': [['R'], ['G'], ['B'], ['B']]
    }
    # 2. Feature columns (Sparse)
    price = feature_column.categorical_column_with_vocabulary_list(
        'price', ['A', 'B', 'C', 'D'])
    # 2. Feature columns (Sparse)
    color = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'])
    # 2. Feature columns (Sparse)
    p_x_c = feature_column.crossed_column([price, color], 16)
    # 2. Feature columns (Dense)
    p_x_c_identity = feature_column.indicator_column(p_x_c)
    # 3. Feature tensor 
    p_x_c_identity_dense_tensor = feature_column.input_layer(featrues,
                                                           [p_x_c_identity])
    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([p_x_c_identity_dense_tensor]))
#test_crossed_column()

def test_embedding():
    tf.set_random_seed(1)
    # 1. Input features
    color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}
    builder = _LazyBuilder(color_data)
    # 2. Feature columns (Sparse)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    color_column_tensor = color_column._get_sparse_tensors(builder)
    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))

    # 2. Feature columns (Dense)
    color_embedding = feature_column.embedding_column(color_column, 4,
                                                      combiner='sum')
    # 3. Feature tensor 
    color_embedding_dense_tensor = feature_column.input_layer(color_data,
                                                              [color_embedding])

    with tf.Session() as session:
        # Embedding needs variables (weights) to do the embedding
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('embedding' + '_' * 40)
        print(session.run([color_embedding_dense_tensor]))

#test_embedding()

def test_shared_embedding_column_with_hash_bucket():
    # 1. Input features
    color_data = {'range': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'id': [[2], [5], [-1], [0]]}
    builder = _LazyBuilder(color_data)
    # 2. Feature columns (Sparse)
    color_column = feature_column.categorical_column_with_hash_bucket(
        'range', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    # 2. Feature columns (Sparse)
    color_column2 = feature_column.categorical_column_with_hash_bucket(
        'id', 7, dtype=tf.int32)
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)
    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))

    # 2. Feature columns (Dense)
    color_column_embed = feature_column.shared_embedding_columns(
        [color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    # 3. Feature tensor 
    color_dense_tensor = feature_column.input_layer(color_data,
                                                    color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run(color_dense_tensor))

#test_shared_embedding_column_with_hash_bucket()

def test_weighted_categorical_column():
    # 1. Input features
    color_data = {'color': [['R'], ['G'], ['B'], ['A']],
                  'weight': [[1.0], [2.0], [4.0], [8.0]]}
    # 2. Feature columns (Sparse)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )
    # 2. Feature columns (Sparse)
    color_weight_categorical_column \
        = feature_column.weighted_categorical_column(color_column, 'weight')
    builder = _LazyBuilder(color_data)
    id_tensor, weight = color_weight_categorical_column._get_sparse_tensors(
        builder)

    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('weighted categorical' + '-' * 40)
        print(session.run([id_tensor]))
        print('-' * 40)
        print(session.run([weight]))

    # 2. Feature columns (Dense)
    weighted_column = feature_column.indicator_column(
        color_weight_categorical_column)
    # 3. Feature tensor 
    weighted_column_dense_tensor = feature_column.input_layer(color_data,
                                                              [weighted_column])
    with tf.Session() as session:
        #session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run([weighted_column_dense_tensor]))


#test_weighted_categorical_column()
