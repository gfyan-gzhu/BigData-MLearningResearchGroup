class Config:
    model = 'model.lstm'
    alstmnet = {
        'text': True,
        'price': True,
        'wavelet': True
    }
    # 数据参数
    csv_file = './data/000001sh.csv'
    # csv_file = './data/dataset1.csv'
    # csv_file = './data/399001sz.csv'
    # csv_file = './data/dataset2.csv'
    # 网络参数
    input_size = 8
    if model == 'model.alstmnet' and alstmnet['text']:
      input_size = 9 # alstmnet增加情绪极值数据
    print(input_size)
    output_size = 2
    cnn_filter = 16
    hidden_size = 64  # LSTM的隐藏层大小，也是输出大小
    num_directions = 2 # 双向lstm
    # lstm_layers = 1  # LSTM的堆叠层数
    dropout_rate = 0  # dropout概率
    time_step = 10 # 这个参数很重要，是设置用前多少天的数据来预测，也是LSTM的time step数，请保证训练数据量大于它

    # 训练参数
    do_train = True
    do_predict = True
    add_train = False  # 是否载入已有模型参数进行增量训练
    shuffle_train_data = False  # 是否对训练数据做shuffle
    use_cuda = True  # 是否使用GPU训练

    train_data_rate = 0.8  # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.2  # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 32
    learning_rate = 1e-2
    epoch = 100  # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 5  # 训练多少epoch，验证集没提升就停掉
    random_seed = 42  # 随机种子，保证可复现