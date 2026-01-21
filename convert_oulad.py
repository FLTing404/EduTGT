"""
OULAD → ContraTGT 数据转换脚本
将 OULAD 教育日志转换为时间二部图格式
"""

import pandas as pd
import numpy as np
import os

def load_oulad_data(data_dir):
    """加载 OULAD 原始数据"""
    print("正在加载 OULAD 数据...")
    
    student_vle = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))
    student_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
    courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
    
    print(f"studentVle: {student_vle.shape}")
    print(f"studentInfo: {student_info.shape}")
    print(f"courses: {courses.shape}")
    
    return student_vle, student_info, courses

def build_node_mapping(student_info, student_vle):
    """构建节点映射：学生节点 + 课程节点"""
    print("\n正在构建节点映射...")
    
    # 1. 学生节点映射（从1开始）
    unique_students = sorted(student_info['id_student'].unique())
    student_to_id = {s: idx + 1 for idx, s in enumerate(unique_students)}
    num_students = len(unique_students)
    
    print(f"学生节点数: {num_students}")
    
    # 2. 课程节点映射（code_module + code_presentation）
    # 从 studentVle 中获取所有唯一的课程组合
    student_vle['course_key'] = student_vle['code_module'].astype(str) + '_' + student_vle['code_presentation'].astype(str)
    unique_courses = sorted(student_vle['course_key'].unique())
    course_to_id = {c: idx + num_students + 1 for idx, c in enumerate(unique_courses)}
    num_courses = len(unique_courses)
    
    print(f"课程节点数: {num_courses}")
    print(f"总节点数: {num_students + num_courses}")
    
    return student_to_id, course_to_id, num_students, num_courses

def get_student_course_labels(student_info):
    """获取 (学生, 课程) 层面的标签"""
    print("\n正在构建标签映射...")
    
    # 创建 (学生, 课程) -> label 的映射
    label_map = {}
    
    for _, row in student_info.iterrows():
        student_id = row['id_student']
        course_key = str(row['code_module']) + '_' + str(row['code_presentation'])
        final_result = row['final_result']
        
        # Pass = 1, Fail/Withdraw = -1
        if final_result == 'Pass' or final_result == 'Distinction':
            label = 1
        else:  # Fail, Withdraw
            label = -1
        
        label_map[(student_id, course_key)] = label
    
    print(f"标签映射数量: {len(label_map)}")
    print(f"正样本 (Pass): {sum(1 for v in label_map.values() if v == 1)}")
    print(f"负样本 (Fail/Withdraw): {sum(1 for v in label_map.values() if v == -1)}")
    
    return label_map

def build_edges(student_vle, student_to_id, course_to_id, label_map):
    """构建边数据"""
    print("\n正在构建边数据...")
    
    edges = []
    
    # 获取最小时间戳用于归一化
    min_date = student_vle['date'].min()
    
    # 统计信息
    skipped_student = 0
    skipped_course = 0
    no_label = 0
    
    for idx, row in student_vle.iterrows():
        student_id = row['id_student']
        course_key = str(row['code_module']) + '_' + str(row['code_presentation'])
        date = row['date']
        
        # 跳过无效的学生
        if student_id not in student_to_id:
            skipped_student += 1
            continue
        
        # 跳过无效的课程
        if course_key not in course_to_id:
            skipped_course += 1
            continue
        
        # 获取节点ID
        u = student_to_id[student_id]  # 学生节点
        i = course_to_id[course_key]   # 课程节点
        
        # 时间戳：使用相对天数（从最小日期开始）
        # 确保时间戳为整数且从1开始
        ts = max(1, int(date - min_date + 1))
        
        # 获取标签（基于最终结果）
        label = label_map.get((student_id, course_key))
        if label is None:
            no_label += 1
            # 如果没有标签，跳过这条边（或者使用默认值1）
            # 这里选择跳过，确保所有边都有明确的标签
            continue
        
        # 边索引（从1开始）
        edge_idx = len(edges) + 1
        
        edges.append({
            'id': edge_idx,
            'u': u,
            'i': i,
            'ts': ts,
            'label': label,
            'idx': edge_idx
        })
    
    edges_df = pd.DataFrame(edges)
    
    # 按时间戳排序
    edges_df = edges_df.sort_values('ts').reset_index(drop=True)
    edges_df['id'] = range(1, len(edges_df) + 1)
    edges_df['idx'] = edges_df['id']
    
    print(f"总边数: {len(edges_df)}")
    print(f"时间戳范围: {edges_df['ts'].min()} - {edges_df['ts'].max()}")
    print(f"正边数: {(edges_df['label'] == 1).sum()}")
    print(f"负边数: {(edges_df['label'] == -1).sum()}")
    if skipped_student > 0 or skipped_course > 0 or no_label > 0:
        print(f"跳过的边: 学生无效={skipped_student}, 课程无效={skipped_course}, 无标签={no_label}")
    
    return edges_df

def compute_course_statistics(student_vle, courses, course_to_id):
    """计算课程统计特征"""
    print("\n正在计算课程统计量...")
    
    # 创建 course_key
    student_vle_copy = student_vle.copy()
    student_vle_copy['course_key'] = student_vle_copy['code_module'].astype(str) + '_' + student_vle_copy['code_presentation'].astype(str)
    courses_copy = courses.copy()
    courses_copy['course_key'] = courses_copy['code_module'].astype(str) + '_' + courses_copy['code_presentation'].astype(str)
    
    course_stats = {}
    
    for course_key in sorted(course_to_id.keys()):
        # 1. 课程总点击量（log1p）
        course_clicks = student_vle_copy[student_vle_copy['course_key'] == course_key]['sum_click'].sum()
        total_clicks_log = np.log1p(course_clicks)
        
        # 2. 课程活跃学生数（log1p）
        active_students = student_vle_copy[student_vle_copy['course_key'] == course_key]['id_student'].nunique()
        active_students_log = np.log1p(active_students)
        
        # 3. 课程周期长度（module_presentation_length）
        course_info = courses_copy[courses_copy['course_key'] == course_key]
        if len(course_info) > 0:
            course_length = course_info.iloc[0]['module_presentation_length']
            if pd.isna(course_length) or course_length <= 0:
                # 如果数据无效，从 VLE 数据推断
                date_range = student_vle_copy[student_vle_copy['course_key'] == course_key]['date']
                if len(date_range) > 0:
                    course_length = date_range.max() - date_range.min() + 1
                else:
                    course_length = 1
        else:
            # 如果没有找到，从数据推断
            date_range = student_vle_copy[student_vle_copy['course_key'] == course_key]['date']
            if len(date_range) > 0:
                course_length = date_range.max() - date_range.min() + 1
            else:
                course_length = 1
        
        course_stats[course_key] = {
            'total_clicks_log': total_clicks_log,
            'active_students_log': active_students_log,
            'course_length': float(course_length)
        }
    
    print(f"  计算了 {len(course_stats)} 个课程的统计量")
    return course_stats

def build_node_features(student_info, num_students, num_courses, course_to_id, course_stats):
    """构建节点特征"""
    print("\n正在构建节点特征...")
    
    # 1. 学生特征编码
    student_features = []
    
    # 获取所有唯一值用于 one-hot 编码
    unique_age_bands = sorted(student_info['age_band'].fillna('Unknown').unique())
    unique_educations = sorted(student_info['highest_education'].fillna('Unknown').unique())
    
    age_band_to_idx = {age: idx for idx, age in enumerate(unique_age_bands)}
    education_to_idx = {edu: idx for idx, edu in enumerate(unique_educations)}
    
    num_age_bands = len(unique_age_bands)
    num_educations = len(unique_educations)
    
    print(f"  age_band 类别数: {num_age_bands}")
    print(f"  education 类别数: {num_educations}")
    
    # 计算 studied_credits 的最大值用于归一化
    max_credits = student_info['studied_credits'].fillna(0).max()
    if max_credits == 0:
        max_credits = 1
    
    # 为每个学生构建特征（按 student_id 排序，确保与节点ID对应）
    for student_id in sorted(student_info['id_student'].unique()):
        student_rows = student_info[student_info['id_student'] == student_id]
        # 如果有多个记录，取第一个（通常一个学生一个记录）
        student_row = student_rows.iloc[0]
        
        # age_band one-hot
        age_band = student_row.get('age_band', 'Unknown')
        if pd.isna(age_band):
            age_band = 'Unknown'
        age_idx = age_band_to_idx.get(age_band, 0)
        age_onehot = np.zeros(num_age_bands)
        age_onehot[age_idx] = 1
        
        # highest_education one-hot
        education = student_row.get('highest_education', 'Unknown')
        if pd.isna(education):
            education = 'Unknown'
        edu_idx = education_to_idx.get(education, 0)
        edu_onehot = np.zeros(num_educations)
        edu_onehot[edu_idx] = 1
        
        # studied_credits (归一化)
        studied_credits = student_row.get('studied_credits', 0)
        if pd.isna(studied_credits):
            studied_credits = 0
        studied_credits_norm = min(studied_credits / max_credits, 1.0)
        
        # disability (0/1)
        disability = student_row.get('disability', 'N')
        disability_val = 1 if str(disability).upper() == 'Y' else 0
        
        # 拼接特征向量（原有特征 + 3个占位符 + 节点类型标识 + 填充维度）
        # 占位符用于与课程特征的统计量维度对齐
        # 填充维度确保特征维度能被多头注意力的头数（4）整除
        feature = np.concatenate([
            age_onehot,
            edu_onehot,
            [studied_credits_norm],
            [disability_val],
            [0.0, 0.0, 0.0],  # 占位符（对应课程的3个统计量）
            [1, 0],  # [is_student=1, is_course=0]
            [0.0]  # 填充维度，使总维度为16（能被4整除）
        ])
        
        student_features.append(feature)
    
    # 确保学生特征数量正确
    if len(student_features) != num_students:
        print(f"警告: 学生特征数 ({len(student_features)}) 与节点数 ({num_students}) 不匹配")
        # 补充缺失的学生特征
        while len(student_features) < num_students:
            base_feature_dim = num_age_bands + num_educations + 2  # 原有特征维度
            feature_dim = base_feature_dim + 3 + 2 + 1  # 加上占位符、节点类型标识和填充维度
            student_features.append(np.concatenate([
                np.zeros(base_feature_dim),
                [0.0, 0.0, 0.0],  # 占位符
                [1, 0],  # 节点类型
                [0.0]  # 填充维度
            ]))
    
    # 2. 课程特征（使用统计量）
    course_features = []
    base_feature_dim = num_age_bands + num_educations + 2  # 原有特征维度
    
    for course_key in sorted(course_to_id.keys()):
        stats = course_stats.get(course_key, {
            'total_clicks_log': 0.0,
            'active_students_log': 0.0,
            'course_length': 1.0
        })
        
        # 构建课程特征：对齐零向量 + 统计量 + 节点类型标识 + 填充维度
        course_feature = np.concatenate([
            np.zeros(base_feature_dim),  # 与学生特征维度对齐（全零）
            [stats['total_clicks_log']],
            [stats['active_students_log']],
            [stats['course_length']],
            [0, 1],  # [is_student=0, is_course=1]
            [0.0]  # 填充维度，使总维度为16（能被4整除）
        ])
        
        course_features.append(course_feature)
    
    # 3. 合并所有节点特征（学生在前，课程在后）
    all_features = student_features + course_features
    
    # 更新特征维度
    feature_dim = len(all_features[0])
    
    print(f"特征维度: {feature_dim}")
    print(f"  学生特征: {base_feature_dim} (原有) + 3 (占位符) + 2 (节点类型) + 1 (填充) = {feature_dim}")
    print(f"  课程特征: {base_feature_dim} (对齐) + 3 (统计量) + 2 (节点类型) + 1 (填充) = {feature_dim}")
    # 验证特征维度能被多头注意力的头数（4）整除
    if feature_dim % 4 != 0:
        print(f"警告: 特征维度 {feature_dim} 不能被4整除，可能导致模型初始化失败！")
    else:
        print(f"✓ 特征维度 {feature_dim} 可以被4整除，符合模型要求")
    print(f"学生特征数: {len(student_features)}")
    print(f"课程特征数: {len(course_features)}")
    print(f"总特征数: {len(all_features)}")
    
    return all_features, feature_dim

def save_contraTGT_format(edges_df, node_features, output_dir):
    """保存为 ContraTGT 格式"""
    print("\n正在保存文件...")
    
    # 1. 保存边数据 ml_oulad.csv
    edges_output_path = os.path.join(output_dir, 'ml_oulad.csv')
    edges_df.to_csv(edges_output_path, index=False)
    print(f"✓ 边数据已保存: {edges_output_path}")
    print(f"  格式: id,u,i,ts,label,idx")
    print(f"  行数: {len(edges_df)}")
    
    # 2. 保存节点特征 oulad.content
    script_dir = os.path.dirname(os.path.abspath(output_dir))
    node_feature_dir = os.path.join(script_dir, 'ContraTGT', 'node_feature')
    os.makedirs(node_feature_dir, exist_ok=True)
    
    node_feature_path = os.path.join(node_feature_dir, 'oulad.content')
    
    # 将特征写入文件（逗号分隔）
    with open(node_feature_path, 'w') as f:
        for feature in node_features:
            feature_str = ','.join([str(x) for x in feature])
            f.write(feature_str + '\n')
    
    print(f"✓ 节点特征已保存: {node_feature_path}")
    print(f"  行数: {len(node_features)}")
    
    return edges_output_path, node_feature_path

def main():
    """主函数"""
    # 路径配置（相对于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    oulad_data_dir = os.path.join(script_dir, 'OULAD-main', 'data')  # OULAD 原始数据目录
    output_dir = os.path.join(script_dir, 'data')  # 输出目录（ContraTGT/data）
    
    # 检查数据目录
    if not os.path.exists(oulad_data_dir):
        print(f"错误: 找不到数据目录 {oulad_data_dir}")
        print("请确保 OULAD 原始数据已下载到该目录")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载数据
    student_vle, student_info, courses = load_oulad_data(oulad_data_dir)
    
    # 2. 构建节点映射
    student_to_id, course_to_id, num_students, num_courses = build_node_mapping(student_info, student_vle)
    
    # 3. 计算课程统计量
    course_stats = compute_course_statistics(student_vle, courses, course_to_id)
    
    # 4. 获取标签映射
    label_map = get_student_course_labels(student_info)
    
    # 5. 构建边数据
    edges_df = build_edges(student_vle, student_to_id, course_to_id, label_map)
    
    # 6. 构建节点特征
    node_features, feature_dim = build_node_features(student_info, num_students, num_courses, 
                                                      course_to_id, course_stats)
    
    # 6. 保存文件
    edges_path, feature_path = save_contraTGT_format(edges_df, node_features, output_dir)
    
    print("\n" + "="*50)
    print("转换完成！")
    print("="*50)
    print(f"\n生成的文件:")
    print(f"  1. {edges_path}")
    print(f"  2. {feature_path}")
    print(f"\n数据统计:")
    print(f"  - 学生节点: {num_students}")
    print(f"  - 课程节点: {num_courses}")
    print(f"  - 总边数: {len(edges_df)}")
    print(f"  - 特征维度: {feature_dim}")
    print(f"\n下一步:")
    print(f"  1. 在 utils.py 中添加 'oulad' 到数据集 choices")
    print(f"  2. 运行预训练: python pretrain.py -d oulad --bs 800 --ctx_sample 30 --tmp_sample 21 --seed 60")
    print(f"  3. 运行训练: python main.py -d oulad --bs 800 --ctx_sample 40 --tmp_sample 31 --seed 60")
    print(f"\n注意: 如果数据量很大，可能需要调整 batch_size 和采样参数")

if __name__ == '__main__':
    main()

