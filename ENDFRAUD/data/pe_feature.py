import numpy as np
from scipy.io import loadmat

from scipy.sparse import csr_matrix, diags


# 设置随机种子以确保初始化一致
def set_random_seed(seed_value):
    np.random.seed(seed_value)


# 定义随机游走矩阵 RW
def random_walk_matrix(adj_matrix):
    degree_vector = np.array(adj_matrix.sum(axis=1)).flatten()

    # 处理度数为0的节点，避免除以0的情况
    degree_vector[degree_vector == 0] = 1

    degree_vector_inv = 1.0 / degree_vector
    D_inv = diags(degree_vector_inv)
    return adj_matrix.dot(D_inv)


# 计算 RWPE 编码
def compute_rwpe(adj_matrix, k_steps=3):
    RW = random_walk_matrix(adj_matrix)

    # 初始化 RWPE 为随机游走矩阵的对角线
    RWPE = RW.diagonal().reshape(-1, 1)  # 确保 RWPE 形状正确

    # 逐步计算随机游走矩阵的幂次 RW^2, RW^3, ..., RW^k
    current_rw = RW.copy()
    for i in range(2, k_steps + 1):
        current_rw = current_rw.dot(RW)  # 使用矩阵乘法逐步更新 RW^i
        RWPE = np.hstack([RWPE, current_rw.diagonal().reshape(-1, 1)])

    return RWPE


# 初始 PE 映射到 d 维特征向量
def initial_pe_mapping(RWPE, d_dim, seed_value):
    # 设置随机种子以确保每次生成相同的 C0 和 c0
    set_random_seed(seed_value)

    k_dim = RWPE.shape[1]
    C0 = np.random.rand(d_dim, k_dim)  # 随机初始化映射矩阵 C0
    c0 = np.random.rand(d_dim)  # 偏置项 c0
    return RWPE.dot(C0.T) + c0


# 示例使用
if __name__ == "__main__":
    yelp = loadmat('data/Amazon.mat')
    yelp_homo = yelp['homo']
    adj_matrix = csr_matrix(yelp_homo)
    k_steps = 3

    # 计算 RWPE
    RWPE = compute_rwpe(adj_matrix, k_steps)

    # 初始 PE 的 d 维映射
    # d_dim = 32  # 设定输出维度 d
    d_dim = 25
    seed_value = 42  # 设置一个固定的随机种子
    p_i_0 = initial_pe_mapping(RWPE, d_dim, seed_value)

    print("初始 PE 特征向量：")
    print(p_i_0)
    print(p_i_0.shape)

    import numpy as np
    from scipy.io import savemat

    # p_i_0 是你计算得到的 PE 特征向量

    # 保存为 .mat 文件
    savemat('pe_features.mat', {'pe_features': p_i_0})
