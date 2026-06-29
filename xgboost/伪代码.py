# 1. 数据加载
data = pd.read_csv('data.csv')

# 2. 准备输入 X 和输出 y
#    X原始: 6层特征 (需从你的1039维描述中提取)
#    y: [log1p(LT99), log1p(LT97), log1p(LT95), dV99, dV97, dV95, init_V]
#    追加物理条件: J, T (用于输入)

# 3. 按 device_id 分组划分
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=data['device_id']))

# 4. 对训练集X进行PCA(64维)，并应用到测试集；对y_logLT和y_dV计算变换参数

# 5. 训练深度编码器 (只使用训练集)
encoder = MolecularEncoder()  # 上述架构
encoder.fit(X_train, y_train[['logLT95', 'init_V']])  # 监督预训练

# 6. 提取训练集和测试集的32维分子表征
F_train = encoder.extract_features(X_train)
F_test = encoder.extract_features(X_test)

# 7. 拼接物理条件 [logJ, 1000/T, logJ*1000/T]
final_train = np.concatenate([F_train, phys_train], axis=1)
final_test = np.concatenate([F_test, phys_test], axis=1)

# 8. 训练7个XGBoost模型 (循环)
for target in ['LT99','LT97','LT95','dV99','dV97','dV95','init_V']:
    model = xgb.XGBRegressor(**params[target])
    model.fit(final_train, y_train[target], 
              eval_set=[(final_val, y_val[target])], 
              early_stopping_rounds=30, verbose=False)
    # 保存模型

# 9. 测试集预测 + 后处理校准
preds = np.array([model.predict(final_test) for model in models]).T
preds_calibrated = apply_physical_constraints(preds)  # 排序与平滑