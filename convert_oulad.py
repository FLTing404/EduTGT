"""OULAD数据转换为ContraTGT格式 - 充分利用所有7个数据文件"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def load_all_oulad_data(data_dir):
    """加载所有7个OULAD数据文件"""
    student_vle = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))
    student_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
    courses = pd.read_csv(os.path.join(data_dir, 'courses.csv'))
    assessments = pd.read_csv(os.path.join(data_dir, 'assessments.csv'))
    student_assessment = pd.read_csv(os.path.join(data_dir, 'studentAssessment.csv'))
    student_registration = pd.read_csv(os.path.join(data_dir, 'studentRegistration.csv'))
    vle = pd.read_csv(os.path.join(data_dir, 'vle.csv'))
    
    return (student_vle, student_info, courses, assessments, 
            student_assessment, student_registration, vle)

def build_node_mapping(student_info, student_vle):
    """构建学生和课程的节点映射"""
    student_vle['course_key'] = student_vle['code_module'].astype(str) + '_' + student_vle['code_presentation'].astype(str)
    
    # 只提取在studentVle中出现的唯一学生
    unique_students = sorted(student_vle['id_student'].unique())
    student_to_id = {s: idx + 1 for idx, s in enumerate(unique_students)}
    num_students = len(unique_students)
    
    unique_courses = sorted(student_vle['course_key'].unique())
    course_to_id = {c: idx + num_students + 1 for idx, c in enumerate(unique_courses)}
    num_courses = len(unique_courses)
    
    return student_to_id, course_to_id, num_students, num_courses

def get_student_course_labels(student_info):
    """获取学生-课程标签映射"""
    label_map = {}
    for _, row in student_info.iterrows():
        student_id = row['id_student']
        course_key = str(row['code_module']) + '_' + str(row['code_presentation'])
        final_result = row['final_result']
        
        if final_result == 'Pass' or final_result == 'Distinction':
            label = 1
        else:
            label = -1
        
        label_map[(student_id, course_key)] = label
    return label_map

def compute_course_progress(student_vle, courses):
    """
    计算课程进度相关特征（方案汇总：课程进度特征）
    - 每门课的起止日期与长度（天）
    - 每个 (学生, 课程) 的「前 1/3 课程参与度」
    """
    student_vle = student_vle.copy()
    student_vle['course_key'] = (
        student_vle['code_module'].astype(str) + '_' + student_vle['code_presentation'].astype(str)
    )
    courses = courses.copy()
    courses['course_key'] = (
        courses['code_module'].astype(str) + '_' + courses['code_presentation'].astype(str)
    )
    
    # 每门课的时间范围（天）与长度
    course_dates = student_vle.groupby('course_key')['date'].agg(['min', 'max']).reset_index()
    course_dates.columns = ['course_key', 'date_min', 'date_max']
    course_dates['length_days'] = course_dates['date_max'] - course_dates['date_min'] + 1
    
    # 用 courses 表的 module_presentation_length 若存在
    if 'module_presentation_length' in courses.columns:
        len_map = courses.drop_duplicates('course_key').set_index('course_key')['module_presentation_length']
        for k in course_dates['course_key']:
            if k in len_map.index and pd.notna(len_map[k]) and len_map[k] > 0:
                course_dates.loc[course_dates['course_key'] == k, 'length_days'] = int(len_map[k])
    
    # 每个 (学生, 课程) 的参与度：前 1/3 时间内的点击占比
    participation_first_third = {}
    for (student_id, course_key), grp in tqdm(student_vle.groupby(['id_student', 'course_key']), desc="课程进度参与度"):
        dates = grp['date'].values
        row = course_dates[course_dates['course_key'] == course_key].iloc[0]
        d_min, d_max = row['date_min'], row['date_max']
        length = max(1, row['length_days'])
        first_third_end = d_min + length / 3.0
        in_first_third = (dates <= first_third_end).sum()
        total = len(dates)
        ratio = in_first_third / total if total > 0 else 0.0
        participation_first_third[(student_id, course_key)] = ratio
    
    return course_dates, participation_first_third


def build_edges(student_vle, student_to_id, course_to_id, label_map, 
                course_dates=None, participation_first_third=None):
    """构建边数据；可选附加课程进度特征（当前周、前1/3参与度）"""
    edges = []
    min_date = student_vle['date'].min()
    student_vle_copy = student_vle.copy()
    student_vle_copy['course_key'] = (
        student_vle_copy['code_module'].astype(str) + '_' + student_vle_copy['code_presentation'].astype(str)
    )
    
    for idx, row in tqdm(student_vle_copy.iterrows(), total=len(student_vle_copy), desc="构建边"):
        student_id = row['id_student']
        course_key = str(row['code_module']) + '_' + str(row['code_presentation'])
        date = row['date']
        
        if student_id not in student_to_id or course_key not in course_to_id:
            continue
        
        u = student_to_id[student_id]
        i = course_to_id[course_key]
        ts = max(1, int(date - min_date + 1))
        
        label = label_map.get((student_id, course_key))
        if label is None:
            continue
        
        week_in_course = 0.0
        part_first_third = 0.0
        if course_dates is not None and participation_first_third is not None:
            cr = course_dates[course_dates['course_key'] == course_key]
            if len(cr) > 0:
                d_min = cr.iloc[0]['date_min']
                length_days = max(1, cr.iloc[0]['length_days'])
                week_in_course = (date - d_min) / 7.0
                week_in_course = max(0.0, min(week_in_course / 52.0, 1.0))  # 归一化到约 0~1（52 周）
            part_first_third = participation_first_third.get((student_id, course_key), 0.0)
        
        edge_idx = len(edges) + 1
        edges.append({
            'id': edge_idx,
            'u': u,
            'i': i,
            'ts': ts,
            'label': label,
            'idx': edge_idx,
            'week_in_course': week_in_course,
            'participation_first_third': part_first_third,
        })
    
    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.sort_values('ts').reset_index(drop=True)
    edges_df['id'] = range(1, len(edges_df) + 1)
    edges_df['idx'] = edges_df['id']
    
    assert edges_df['idx'].min() == 1
    assert edges_df['idx'].max() == len(edges_df)
    assert len(edges_df['idx'].unique()) == len(edges_df)
    
    return edges_df

def extract_student_assessment_features(student_assessment, assessments, student_ids):
    """从studentAssessment和assessments提取学生评估特征"""
    # 合并数据获取课程信息
    student_assessment = student_assessment.merge(
        assessments[['id_assessment', 'code_module', 'code_presentation']], 
        on='id_assessment', how='left'
    )
    student_assessment['course_key'] = (
        student_assessment['code_module'].astype(str) + '_' + 
        student_assessment['code_presentation'].astype(str)
    )
    
    student_assessment_features = {}
    
    for student_id in tqdm(student_ids, desc="学生评估特征"):
        student_assessments = student_assessment[student_assessment['id_student'] == student_id]
        
        if len(student_assessments) == 0:
            # 没有评估数据
            student_assessment_features[student_id] = {
                'avg_score': 0.0,
                'std_score': 0.0,
                'num_assessments': 0.0,
                'has_banked': 0.0,
                'avg_submission_delay': 0.0
            }
        else:
            scores = student_assessments['score'].dropna()
            avg_score = scores.mean() / 100.0 if len(scores) > 0 else 0.0  # 归一化到0-1
            std_score = scores.std() / 100.0 if len(scores) > 1 else 0.0
            
            num_assessments = np.log1p(len(student_assessments))
            has_banked = 1.0 if student_assessments['is_banked'].sum() > 0 else 0.0
            
            # 计算提交延迟（相对于评估日期）
            student_assessments_merged = student_assessments.merge(
                assessments[['id_assessment', 'date']], 
                on='id_assessment', how='left'
            )
            student_assessments_merged = student_assessments_merged[
                student_assessments_merged['date'].notna() & 
                student_assessments_merged['date_submitted'].notna()
            ]
            if len(student_assessments_merged) > 0:
                delays = student_assessments_merged['date_submitted'] - student_assessments_merged['date']
                avg_delay = delays.mean() if len(delays) > 0 else 0.0
                avg_delay_norm = avg_delay / 100.0  # 归一化
            else:
                avg_delay_norm = 0.0
            
            student_assessment_features[student_id] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'num_assessments': num_assessments,
                'has_banked': has_banked,
                'avg_submission_delay': avg_delay_norm
            }
    
    return student_assessment_features

def extract_student_registration_features(student_registration, student_ids):
    """从studentRegistration提取学生注册特征"""
    student_registration['course_key'] = (
        student_registration['code_module'].astype(str) + '_' + 
        student_registration['code_presentation'].astype(str)
    )
    
    registration_features = {}
    
    for student_id in tqdm(student_ids, desc="学生注册特征"):
        student_regs = student_registration[student_registration['id_student'] == student_id]
        
        if len(student_regs) == 0:
            registration_features[student_id] = {
                'avg_registration_date': 0.0,
                'has_unregistration': 0.0,
                'num_courses_registered': 0.0
            }
        else:
            # 注册时间特征（归一化）
            reg_dates = student_regs['date_registration'].fillna(0)
            avg_reg_date = reg_dates.mean() / 200.0  # 归一化（假设范围在-200到200）
            
            # 是否有退课
            has_unreg = 1.0 if student_regs['date_unregistration'].notna().any() else 0.0
            
            # 注册的课程数
            num_courses = np.log1p(len(student_regs))
            
            registration_features[student_id] = {
                'avg_registration_date': avg_reg_date,
                'has_unregistration': has_unreg,
                'num_courses_registered': num_courses
            }
    
    return registration_features

def extract_student_vle_features(student_vle, student_ids):
    """从studentVle提取学生VLE交互特征"""
    student_vle['course_key'] = (
        student_vle['code_module'].astype(str) + '_' + 
        student_vle['code_presentation'].astype(str)
    )
    
    vle_features = {}
    
    for student_id in tqdm(student_ids, desc="学生VLE特征"):
        student_vle_data = student_vle[student_vle['id_student'] == student_id]
        
        if len(student_vle_data) == 0:
            vle_features[student_id] = {
                'total_clicks_log': 0.0,
                'unique_days': 0.0,
                'avg_clicks_per_day': 0.0,
                'unique_sites': 0.0
            }
        else:
            total_clicks = student_vle_data['sum_click'].sum()
            total_clicks_log = np.log1p(total_clicks)
            
            unique_days = student_vle_data['date'].nunique()
            unique_days_norm = unique_days / 300.0  # 归一化（假设最多300天）
            
            avg_clicks_per_day = total_clicks / unique_days if unique_days > 0 else 0.0
            avg_clicks_per_day_log = np.log1p(avg_clicks_per_day)
            
            unique_sites = student_vle_data['id_site'].nunique()
            unique_sites_log = np.log1p(unique_sites)
            
            vle_features[student_id] = {
                'total_clicks_log': total_clicks_log,
                'unique_days': unique_days_norm,
                'avg_clicks_per_day': avg_clicks_per_day_log,
                'unique_sites': unique_sites_log
            }
    
    return vle_features

def extract_course_assessment_features(assessments, course_keys):
    """从assessments提取课程评估特征"""
    assessments['course_key'] = (
        assessments['code_module'].astype(str) + '_' + 
        assessments['code_presentation'].astype(str)
    )
    
    # 评估类型 - 固定所有可能的类型
    all_assessment_types = ['TMA', 'CMA', 'Exam']  # 根据数据固定类型
    type_to_idx = {t: idx for idx, t in enumerate(all_assessment_types)}
    num_types = len(all_assessment_types)
    
    course_assessment_features = {}
    
    for course_key in tqdm(course_keys, desc="课程评估特征"):
        course_assessments = assessments[assessments['course_key'] == course_key]
        
        if len(course_assessments) == 0:
            # 创建零特征
            type_counts = np.zeros(num_types)
            num_assessments = 0.0
            avg_weight = 0.0
        else:
            # 评估类型计数（归一化）
            type_counts = np.zeros(num_types)
            for _, row in course_assessments.iterrows():
                a_type = row['assessment_type']
                if a_type in type_to_idx:
                    type_counts[type_to_idx[a_type]] += 1
            type_counts = type_counts / max(type_counts.sum(), 1)  # 归一化
            
            num_assessments = np.log1p(len(course_assessments))
            
            weights = course_assessments['weight'].fillna(0)
            avg_weight = weights.mean() / 100.0 if len(weights) > 0 else 0.0  # 归一化
        
        course_assessment_features[course_key] = {
            'assessment_type_dist': type_counts,
            'num_assessments': num_assessments,
            'avg_weight': avg_weight
        }
    
    return course_assessment_features, all_assessment_types

def extract_course_vle_features(vle, course_keys):
    """从vle提取课程VLE资源特征"""
    vle['course_key'] = (
        vle['code_module'].astype(str) + '_' + 
        vle['code_presentation'].astype(str)
    )
    
    # 主要活动类型
    main_activity_types = ['resource', 'oucontent', 'url', 'subpage', 'forumng', 'quiz']
    type_to_idx = {t: idx for idx, t in enumerate(main_activity_types)}
    
    course_vle_features = {}
    
    for course_key in tqdm(course_keys, desc="课程VLE特征"):
        course_vle = vle[vle['course_key'] == course_key]
        
        if len(course_vle) == 0:
            activity_dist = np.zeros(len(main_activity_types))
            total_resources = 0.0
        else:
            # 活动类型分布（归一化）
            activity_dist = np.zeros(len(main_activity_types))
            for _, row in course_vle.iterrows():
                a_type = row['activity_type']
                if a_type in type_to_idx:
                    activity_dist[type_to_idx[a_type]] += 1
            activity_dist = activity_dist / max(activity_dist.sum(), 1)  # 归一化
            
            total_resources = np.log1p(len(course_vle))
        
        course_vle_features[course_key] = {
            'activity_type_dist': activity_dist,
            'total_resources': total_resources
        }
    
    return course_vle_features

def compute_course_statistics(student_vle, courses, course_to_id):
    """计算课程统计特征"""
    student_vle_copy = student_vle.copy()
    student_vle_copy['course_key'] = (
        student_vle_copy['code_module'].astype(str) + '_' + 
        student_vle_copy['code_presentation'].astype(str)
    )
    courses_copy = courses.copy()
    courses_copy['course_key'] = (
        courses_copy['code_module'].astype(str) + '_' + 
        courses_copy['code_presentation'].astype(str)
    )
    
    course_stats = {}
    for course_key in tqdm(sorted(course_to_id.keys()), desc="课程统计"):
        course_clicks = student_vle_copy[student_vle_copy['course_key'] == course_key]['sum_click'].sum()
        total_clicks_log = np.log1p(course_clicks)
        
        active_students = student_vle_copy[student_vle_copy['course_key'] == course_key]['id_student'].nunique()
        active_students_log = np.log1p(active_students)
        
        course_info = courses_copy[courses_copy['course_key'] == course_key]
        if len(course_info) > 0:
            course_length = course_info.iloc[0]['module_presentation_length']
            if pd.isna(course_length) or course_length <= 0:
                date_range = student_vle_copy[student_vle_copy['course_key'] == course_key]['date']
                course_length = date_range.max() - date_range.min() + 1 if len(date_range) > 0 else 1
        else:
            date_range = student_vle_copy[student_vle_copy['course_key'] == course_key]['date']
            course_length = date_range.max() - date_range.min() + 1 if len(date_range) > 0 else 1
        
        course_length_norm = course_length / 300.0  # 归一化
        
        course_stats[course_key] = {
            'total_clicks_log': total_clicks_log,
            'active_students_log': active_students_log,
            'course_length': course_length_norm
        }
    
    return course_stats

def build_node_features(student_info, num_students, num_courses, course_to_id, 
                       course_stats, student_assessment_features, 
                       student_registration_features, student_vle_features,
                       course_assessment_features, course_vle_features):
    """构建节点特征 - 充分利用所有数据文件"""
    student_features = []
    
    # 学生基础特征（从studentInfo）
    unique_age_bands = sorted(student_info['age_band'].fillna('Unknown').unique())
    unique_educations = sorted(student_info['highest_education'].fillna('Unknown').unique())
    unique_genders = sorted(student_info['gender'].fillna('Unknown').unique())
    unique_regions = sorted(student_info['region'].fillna('Unknown').unique())
    unique_imd_bands = sorted(student_info['imd_band'].fillna('Unknown').unique())
    
    age_band_to_idx = {age: idx for idx, age in enumerate(unique_age_bands)}
    education_to_idx = {edu: idx for idx, edu in enumerate(unique_educations)}
    gender_to_idx = {g: idx for idx, g in enumerate(unique_genders)}
    region_to_idx = {r: idx for idx, r in enumerate(unique_regions)}
    imd_band_to_idx = {imd: idx for idx, imd in enumerate(unique_imd_bands)}
    
    num_age_bands = len(unique_age_bands)
    num_educations = len(unique_educations)
    num_genders = len(unique_genders)
    num_regions = len(unique_regions)
    num_imd_bands = len(unique_imd_bands)
    
    max_credits = student_info['studied_credits'].fillna(0).max()
    if max_credits == 0:
        max_credits = 1
    
    max_prev_attempts = student_info['num_of_prev_attempts'].fillna(0).max()
    if max_prev_attempts == 0:
        max_prev_attempts = 1
    
    # 构建学生特征
    for student_id in tqdm(sorted(student_info['id_student'].unique()), desc="学生特征"):
        student_rows = student_info[student_info['id_student'] == student_id]
        
        # 分类特征 - 取最常见的值
        age_band = student_rows['age_band'].fillna('Unknown').mode()
        age_band = age_band.iloc[0] if len(age_band) > 0 else 'Unknown'
        age_idx = age_band_to_idx.get(age_band, 0)
        age_onehot = np.zeros(num_age_bands)
        age_onehot[age_idx] = 1
        
        education = student_rows['highest_education'].fillna('Unknown').mode()
        education = education.iloc[0] if len(education) > 0 else 'Unknown'
        edu_idx = education_to_idx.get(education, 0)
        edu_onehot = np.zeros(num_educations)
        edu_onehot[edu_idx] = 1
        
        gender = student_rows['gender'].fillna('Unknown').mode()
        gender = gender.iloc[0] if len(gender) > 0 else 'Unknown'
        gender_idx = gender_to_idx.get(gender, 0)
        gender_onehot = np.zeros(num_genders)
        gender_onehot[gender_idx] = 1
        
        region = student_rows['region'].fillna('Unknown').mode()
        region = region.iloc[0] if len(region) > 0 else 'Unknown'
        region_idx = region_to_idx.get(region, 0)
        region_onehot = np.zeros(num_regions)
        region_onehot[region_idx] = 1
        
        imd_band = student_rows['imd_band'].fillna('Unknown').mode()
        imd_band = imd_band.iloc[0] if len(imd_band) > 0 else 'Unknown'
        imd_idx = imd_band_to_idx.get(imd_band, 0)
        imd_onehot = np.zeros(num_imd_bands)
        imd_onehot[imd_idx] = 1
        
        # 数值特征 - 取平均值
        studied_credits = student_rows['studied_credits'].fillna(0).mean()
        studied_credits_norm = min(studied_credits / max_credits, 1.0)
        
        num_prev_attempts = student_rows['num_of_prev_attempts'].fillna(0).mean()
        num_prev_attempts_norm = min(num_prev_attempts / max_prev_attempts, 1.0)
        
        # 布尔特征
        disability = student_rows['disability'].fillna('N')
        disability_val = 1.0 if (disability == 'Y').any() else 0.0
        
        # 从其他数据文件提取的特征
        assessment_feat = student_assessment_features.get(student_id, {
            'avg_score': 0.0, 'std_score': 0.0, 'num_assessments': 0.0,
            'has_banked': 0.0, 'avg_submission_delay': 0.0
        })
        
        registration_feat = student_registration_features.get(student_id, {
            'avg_registration_date': 0.0, 'has_unregistration': 0.0,
            'num_courses_registered': 0.0
        })
        
        vle_feat = student_vle_features.get(student_id, {
            'total_clicks_log': 0.0, 'unique_days': 0.0,
            'avg_clicks_per_day': 0.0, 'unique_sites': 0.0
        })
        
        # 组合所有特征
        feature = np.concatenate([
            age_onehot,                    # age_band one-hot
            edu_onehot,                     # education one-hot
            gender_onehot,                  # gender one-hot
            region_onehot,                  # region one-hot
            imd_onehot,                     # imd_band one-hot
            [studied_credits_norm],         # studied_credits
            [num_prev_attempts_norm],      # num_of_prev_attempts
            [disability_val],               # disability
            [assessment_feat['avg_score']], # 评估平均成绩
            [assessment_feat['std_score']], # 评估成绩标准差
            [assessment_feat['num_assessments']], # 评估数量
            [assessment_feat['has_banked']], # 是否使用banked
            [assessment_feat['avg_submission_delay']], # 平均提交延迟
            [registration_feat['avg_registration_date']], # 平均注册时间
            [registration_feat['has_unregistration']], # 是否有退课
            [registration_feat['num_courses_registered']], # 注册课程数
            [vle_feat['total_clicks_log']], # VLE总点击量
            [vle_feat['unique_days']],      # VLE交互天数
            [vle_feat['avg_clicks_per_day']], # 平均每日点击量
            [vle_feat['unique_sites']],      # 唯一站点数
            [1.0, 0.0],                     # 节点类型标识（学生）
            [0.0]                           # 填充
        ])
        student_features.append(feature)
    
    # 确保学生特征数量正确
    if len(student_features) != num_students:
        base_feature_dim = len(student_features[0]) if len(student_features) > 0 else 0
        while len(student_features) < num_students:
            student_features.append(np.zeros(base_feature_dim))
    
    # 获取学生特征的标准维度
    if len(student_features) == 0:
        raise ValueError("学生特征列表为空！")
    student_feature_dim = len(student_features[0])
    
    # 确保所有学生特征维度一致
    for i in range(len(student_features)):
        if len(student_features[i]) != student_feature_dim:
            if len(student_features[i]) < student_feature_dim:
                student_features[i] = np.concatenate([
                    student_features[i], 
                    np.zeros(student_feature_dim - len(student_features[i]))
                ])
            else:
                student_features[i] = student_features[i][:student_feature_dim]
    
    # 构建课程特征
    course_features = []
    for course_key in tqdm(sorted(course_to_id.keys()), desc="课程特征"):
        stats = course_stats.get(course_key, {
            'total_clicks_log': 0.0,
            'active_students_log': 0.0,
            'course_length': 0.0
        })
        
        assessment_feat = course_assessment_features.get(course_key, {
            'assessment_type_dist': np.array([]),
            'num_assessments': 0.0,
            'avg_weight': 0.0
        })
        
        vle_feat = course_vle_features.get(course_key, {
            'activity_type_dist': np.array([]),
            'total_resources': 0.0
        })
        
        # 获取特征分布（确保维度固定）
        assessment_type_dist = assessment_feat.get('assessment_type_dist', np.array([]))
        if len(assessment_type_dist) == 0:
            assessment_type_dist = np.zeros(3)  # TMA, CMA, Exam
        elif len(assessment_type_dist) != 3:
            # 确保是3维
            if len(assessment_type_dist) < 3:
                assessment_type_dist = np.concatenate([assessment_type_dist, np.zeros(3 - len(assessment_type_dist))])
            else:
                assessment_type_dist = assessment_type_dist[:3]
        
        activity_type_dist = vle_feat.get('activity_type_dist', np.array([]))
        if len(activity_type_dist) == 0:
            activity_type_dist = np.zeros(6)  # 主要活动类型
        elif len(activity_type_dist) != 6:
            # 确保是6维
            if len(activity_type_dist) < 6:
                activity_type_dist = np.concatenate([activity_type_dist, np.zeros(6 - len(activity_type_dist))])
            else:
                activity_type_dist = activity_type_dist[:6]
        
        # 学生特征结构分析：
        # 1. 分类特征（one-hot）：num_age_bands + num_educations + num_genders + num_regions + num_imd_bands
        # 2. 数值特征：studied_credits_norm (1) + num_prev_attempts_norm (1) + disability_val (1) = 3维
        # 3. 评估特征：avg_score (1) + std_score (1) + num_assessments (1) + has_banked (1) + avg_submission_delay (1) = 5维
        # 4. 注册特征：avg_registration_date (1) + has_unregistration (1) + num_courses_registered (1) = 3维
        # 5. VLE特征：total_clicks_log (1) + unique_days (1) + avg_clicks_per_day (1) + unique_sites (1) = 4维
        # 6. 节点类型标识：[1.0, 0.0] = 2维
        # 7. 填充：[0.0] = 1维
        
        student_categorical_dim = num_age_bands + num_educations + num_genders + num_regions + num_imd_bands
        student_numerical_dim = 3  # studied_credits + num_prev_attempts + disability
        student_assessment_dim = 5  # 5个评估相关特征
        student_registration_dim = 3  # 3个注册相关特征
        student_vle_dim = 4  # 4个VLE相关特征
        student_type_dim = 2  # 节点类型标识
        student_padding_dim = 1  # 填充
        
        # 课程特征结构（完全匹配学生特征结构）：
        # 1. 分类特征部分：全0（对齐学生分类特征维度）
        # 2. 数值特征部分：用课程基础统计信息填充（3维）
        # 3. 评估特征部分：用课程评估信息填充（5维：3维评估类型分布 + 评估数量 + 平均权重）
        # 4. 注册特征部分：用课程相关统计填充（3维：可以用课程统计信息）
        # 5. VLE特征部分：用课程VLE信息填充（4维：6维活动类型分布压缩到4维，或使用其他VLE统计）
        # 6. 节点类型标识：[0.0, 1.0] = 2维
        # 7. 填充：[0.0] = 1维
        
        # 5. VLE特征部分（用课程VLE信息，4维）
        # activity_type_dist是6维，但VLE特征部分只有4维，取前4维主要类型
        # 剩余2维信息可以放在注册特征部分
        if len(activity_type_dist) >= 4:
            vle_part = activity_type_dist[:4]
            # 剩余2维可以放在注册特征部分
            activity_remaining = activity_type_dist[4:6] if len(activity_type_dist) >= 6 else np.zeros(2)
        else:
            # 如果不足4维，填充到4维
            vle_part = np.concatenate([activity_type_dist, np.zeros(4 - len(activity_type_dist))])
            activity_remaining = np.zeros(2)
        
        # 构建课程特征，完全匹配学生特征结构
        course_feature = np.concatenate([
            # 1. 分类特征部分（全0，对齐学生分类特征）
            np.zeros(student_categorical_dim),
            
            # 2. 数值特征部分（用课程基础统计信息）
            [stats['total_clicks_log']],         # 课程总点击量（替代studied_credits位置）
            [stats['active_students_log']],       # 活跃学生数（替代num_prev_attempts位置）
            [stats['course_length']],            # 课程长度（替代disability位置）
            
            # 3. 评估特征部分（用课程评估信息，5维）
            assessment_type_dist[:3],            # 评估类型分布的前3维（TMA, CMA, Exam）
            [assessment_feat.get('num_assessments', 0.0)], # 评估数量
            [assessment_feat.get('avg_weight', 0.0)], # 平均权重
            
            # 4. 注册特征部分（用课程相关统计，3维）
            # 使用活动类型分布的剩余2维 + VLE资源总数
            activity_remaining[:2],              # 活动类型分布的剩余2维
            [vle_feat.get('total_resources', 0.0) / 100.0],  # VLE资源总数（归一化）
            
            # 5. VLE特征部分（用课程VLE信息，4维）
            vle_part,
            
            # 6. 节点类型标识
            [0.0, 1.0],                          # 节点类型标识（课程）
            
            # 7. 填充
            [0.0]  # 填充
        ])
        
        # 确保维度完全匹配学生特征
        if len(course_feature) != student_feature_dim:
            if len(course_feature) < student_feature_dim:
                # 如果维度不足，用0填充
                course_feature = np.concatenate([
                    course_feature,
                    np.zeros(student_feature_dim - len(course_feature))
                ])
            else:
                # 如果维度超出，截断（不应该发生，但为了安全）
                course_feature = course_feature[:student_feature_dim]
        
        course_features.append(course_feature)
    
    all_features = student_features + course_features
    
    # 检查并统一所有特征的维度
    if len(all_features) == 0:
        return all_features, 0
    
    # 获取标准维度（使用第一个特征）
    standard_dim = len(all_features[0])
    
    # 确保所有特征维度一致
    for i in range(len(all_features)):
        current_dim = len(all_features[i])
        if current_dim != standard_dim:
            # 如果维度不一致，进行填充或截断
            if current_dim < standard_dim:
                # 填充
                padding = np.zeros(standard_dim - current_dim)
                all_features[i] = np.concatenate([all_features[i], padding])
            else:
                # 截断（不应该发生，但为了安全）
                all_features[i] = all_features[i][:standard_dim]
    
    feature_dim = standard_dim
    
    # 确保特征维度能被4整除
    if feature_dim % 4 != 0:
        padding = 4 - (feature_dim % 4)
        for i in range(len(all_features)):
            all_features[i] = np.concatenate([all_features[i], np.zeros(padding)])
        feature_dim = len(all_features[0])
    
    # 最终验证：确保所有特征维度完全一致
    for i, feat in enumerate(all_features):
        if len(feat) != feature_dim:
            raise ValueError(f"特征 {i} 的维度 {len(feat)} 与标准维度 {feature_dim} 不一致！")
    
    return all_features, feature_dim

def remap_node_ids_for_contraTGT(edges_df):
    """重映射节点ID以符合ContraTGT格式"""
    all_node_ids = []
    seen_ids = set()
    
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="重映射节点ID"):
        u = int(row['u'])
        i = int(row['i'])
        if u not in seen_ids:
            all_node_ids.append(u)
            seen_ids.add(u)
        if i not in seen_ids:
            all_node_ids.append(i)
            seen_ids.add(i)
    
    node_id_map = {original_id: new_id for new_id, original_id in enumerate(all_node_ids)}
    return node_id_map, len(all_node_ids)

def save_edge_features_and_pairs(edges_df, output_dir):
    """
    方案汇总：学生-课程去重 & 课程进度特征
    - 边特征：与 ml_oulad.csv 同序的 week_in_course, participation_first_third
    - 按 (u,i) 聚合的 pair 表：每对只一行，用于「每对只预测一次」训练
    """
    if 'week_in_course' not in edges_df.columns or 'participation_first_third' not in edges_df.columns:
        return None, None
    edge_feature_path = os.path.join(output_dir, 'ml_oulad_edge_feature.csv')
    ef = edges_df[['week_in_course', 'participation_first_third']]
    ef.to_csv(edge_feature_path, index=False)
    pair_path = os.path.join(output_dir, 'ml_oulad_pairs.csv')
    agg = edges_df.groupby(['u', 'i']).agg(
        label=('label', 'first'),
        ts_last=('ts', 'max'),
        edge_count=('idx', 'count'),
        week_in_course_avg=('week_in_course', 'mean'),
        participation_first_third_avg=('participation_first_third', 'mean'),
    ).reset_index()
    agg.to_csv(pair_path, index=False)
    return edge_feature_path, pair_path


def save_contraTGT_format(edges_df, node_features, output_dir, student_to_id, course_to_id, node_feature_dir=None):
    """保存为ContraTGT格式"""
    # 边文件：只保留 ContraTGT 需要的列（id,u,i,ts,label,idx）
    edges_output_path = os.path.join(output_dir, 'ml_oulad.csv')
    out_cols = ['id', 'u', 'i', 'ts', 'label', 'idx']
    edges_df[out_cols].to_csv(edges_output_path, index=False)
    
    node_id_map, num_remapped_nodes = remap_node_ids_for_contraTGT(edges_df)
    reverse_map = {v: k for k, v in node_id_map.items()}
    
    original_to_feature_idx = {}
    feature_idx = 0
    
    for student_id in sorted(student_to_id.keys()):
        original_to_feature_idx[student_to_id[student_id]] = feature_idx
        feature_idx += 1
    
    for course_key in sorted(course_to_id.keys()):
        original_to_feature_idx[course_to_id[course_key]] = feature_idx
        feature_idx += 1
    
    # 获取标准特征维度
    if len(node_features) == 0:
        raise ValueError("节点特征列表为空！")
    standard_feature_dim = len(node_features[0])
    
    # 确保所有特征维度一致
    for i, feat in enumerate(node_features):
        if len(feat) != standard_feature_dim:
            # 如果维度不一致，进行填充或截断
            if len(feat) < standard_feature_dim:
                node_features[i] = np.concatenate([feat, np.zeros(standard_feature_dim - len(feat))])
            else:
                node_features[i] = feat[:standard_feature_dim]
    
    remapped_features = []
    for new_id in tqdm(range(num_remapped_nodes), desc="重映射特征"):
        original_id = reverse_map[new_id]
        if original_id in original_to_feature_idx:
            feature = node_features[original_to_feature_idx[original_id]]
            # 确保维度正确
            if len(feature) != standard_feature_dim:
                if len(feature) < standard_feature_dim:
                    feature = np.concatenate([feature, np.zeros(standard_feature_dim - len(feature))])
                else:
                    feature = feature[:standard_feature_dim]
            remapped_features.append(feature)
        else:
            # 使用标准维度的零向量
            remapped_features.append(np.zeros(standard_feature_dim))
    
    # 节点特征文件也保存在同一个输出目录
    if node_feature_dir is None:
        node_feature_dir = output_dir
    
    os.makedirs(node_feature_dir, exist_ok=True)
    node_feature_path = os.path.join(node_feature_dir, 'oulad.content')
    with open(node_feature_path, 'w') as f:
        for feature in tqdm(remapped_features, desc="写入节点特征"):
            f.write(','.join([str(x) for x in feature]) + '\n')
    
    return edges_output_path, node_feature_path

def main():
    import argparse
    
    print("=" * 80)
    print("开始转换 OULAD 数据（使用所有7个数据文件）...")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description='OULAD 数据转换脚本')
    parser.add_argument('--sample_ratio', type=float, default=1.0, 
                       help='数据采样比例 (0.0-1.0)')
    parser.add_argument('--max_edges', type=int, default=None,
                       help='最大边数限制')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 源码文件夹: code/data/OULAD-main，CSV 在 OULAD-main/data 下
    oulad_data_dir = os.path.join(script_dir, 'data', 'OULAD-main', 'data')

    # 转换结果输出到 code/data/all_data
    if args.sample_ratio == 1.0:
        sample_ratio_str = '1'
    else:
        sample_ratio_str = str(args.sample_ratio)
    output_dir = os.path.join(script_dir, 'data', 'all_data', f'data_{sample_ratio_str}')
    node_feature_dir = output_dir  # 节点特征文件也放在同一个文件夹里
    
    if not os.path.exists(oulad_data_dir):
        print(f"错误: 找不到数据目录 {oulad_data_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载所有数据文件
    print("\n加载所有数据文件...")
    (student_vle, student_info, courses, assessments, 
     student_assessment, student_registration, vle) = load_all_oulad_data(oulad_data_dir)
    print(f"  studentVle: {len(student_vle):,} 行")
    print(f"  studentInfo: {len(student_info):,} 行")
    print(f"  courses: {len(courses)} 行")
    print(f"  assessments: {len(assessments)} 行")
    print(f"  studentAssessment: {len(student_assessment):,} 行")
    print(f"  studentRegistration: {len(student_registration):,} 行")
    print(f"  vle: {len(vle)} 行")
    
    # 构建节点映射
    print("\n构建节点映射...")
    student_to_id, course_to_id, num_students, num_courses = build_node_mapping(student_info, student_vle)
    print(f"  学生节点数: {num_students:,}")
    print(f"  课程节点数: {num_courses}")
    
    # 提取特征
    print("\n提取特征...")
    print("  提取学生评估特征...")
    student_ids = list(student_to_id.keys())
    student_assessment_features = extract_student_assessment_features(
        student_assessment, assessments, student_ids
    )
    
    print("  提取学生注册特征...")
    student_registration_features = extract_student_registration_features(
        student_registration, student_ids
    )
    
    print("  提取学生VLE特征...")
    student_vle_features = extract_student_vle_features(student_vle, student_ids)
    
    print("  提取课程评估特征...")
    course_keys = list(course_to_id.keys())
    course_assessment_features, assessment_types = extract_course_assessment_features(
        assessments, course_keys
    )
    
    print("  提取课程VLE特征...")
    course_vle_features = extract_course_vle_features(vle, course_keys)
    
    print("  计算课程统计特征...")
    course_stats = compute_course_statistics(student_vle, courses, course_to_id)
    
    # 课程进度特征（方案汇总：课程进度）
    print("\n计算课程进度特征...")
    course_dates, participation_first_third = compute_course_progress(student_vle, courses)
    
    # 构建边和标签（含 week_in_course, participation_first_third）
    print("\n构建边数据...")
    label_map = get_student_course_labels(student_info)
    edges_df = build_edges(
        student_vle, student_to_id, course_to_id, label_map,
        course_dates=course_dates, participation_first_third=participation_first_third,
    )
    print(f"  初始边数: {len(edges_df):,}")
    
    # 数据采样
    original_edge_count = len(edges_df)
    if args.sample_ratio < 1.0:
        edges_df = edges_df.sort_values('ts').reset_index(drop=True)
        sample_size = int(len(edges_df) * args.sample_ratio)
        edges_df = edges_df.head(sample_size).reset_index(drop=True)
        print(f"  数据采样: {original_edge_count:,} -> {len(edges_df):,} 条边 ({args.sample_ratio*100:.1f}%)")
    
    if args.max_edges is not None and len(edges_df) > args.max_edges:
        edges_df = edges_df.sort_values('ts').reset_index(drop=True)
        edges_df = edges_df.head(args.max_edges).reset_index(drop=True)
    
    if len(edges_df) < original_edge_count or edges_df['idx'].min() != 1 or (edges_df['idx'].max() - edges_df['idx'].min() + 1) != len(edges_df):
        edges_df = edges_df.sort_values('ts').reset_index(drop=True)
        edges_df['id'] = range(1, len(edges_df) + 1)
        edges_df['idx'] = edges_df['id']
    
    if len(edges_df) < original_edge_count:
        used_students = set(edges_df['u'].unique())
        used_courses = set(edges_df['i'].unique())
        
        reverse_student_map = {v: k for k, v in student_to_id.items()}
        reverse_course_map = {v: k for k, v in course_to_id.items()}
        
        used_student_ids = {reverse_student_map[uid] for uid in used_students if uid in reverse_student_map}
        used_course_keys = {reverse_course_map[cid] for cid in used_courses if cid in reverse_course_map}
        
        student_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(used_students), 1)}
        course_id_map = {old_id: new_id + len(used_students) for new_id, old_id in enumerate(sorted(used_courses), 1)}
        
        edges_df['u'] = edges_df['u'].map(student_id_map)
        edges_df['i'] = edges_df['i'].map(course_id_map)
        
        num_students = len(used_students)
        num_courses = len(used_courses)
        
        student_to_id = {sid: student_id_map[old_id] for sid, old_id in student_to_id.items() if old_id in used_students}
        course_to_id = {ckey: course_id_map[old_id] for ckey, old_id in course_to_id.items() if old_id in used_courses}
        student_info = student_info[student_info['id_student'].isin(used_student_ids)].copy()
    
    # 构建节点特征
    print("\n构建节点特征...")
    used_student_ids = set(student_to_id.keys())
    student_info_filtered = student_info[student_info['id_student'].isin(used_student_ids)].copy()
    
    node_features, feature_dim = build_node_features(
        student_info_filtered, num_students, num_courses, course_to_id, 
        course_stats, student_assessment_features, student_registration_features,
        student_vle_features, course_assessment_features, course_vle_features
    )
    
    # 保存
    print("\n保存数据...")
    edges_path, feature_path = save_contraTGT_format(
        edges_df, node_features, output_dir,
        student_to_id, course_to_id, node_feature_dir
    )
    edge_feat_path, pair_path = save_edge_features_and_pairs(edges_df, output_dir)
    if edge_feat_path:
        print(f"边特征(课程进度): {edge_feat_path}")
    if pair_path:
        print(f"学生-课程对(去重用): {pair_path}")
    
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"边数据: {edges_path}")
    print(f"节点特征: {feature_path}")
    print(f"总边数: {len(edges_df):,}")
    print(f"节点数: {num_students + num_courses:,} (学生: {num_students:,}, 课程: {num_courses})")
    print(f"特征维度: {feature_dim}")
    print(f"\n特征组成:")
    print(f"  学生特征: 基础信息 + 评估特征 + 注册特征 + VLE特征")
    print(f"  课程特征: 统计特征 + 评估特征 + VLE资源特征")

if __name__ == '__main__':
    main()
