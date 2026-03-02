"""
OULAD 数据转换 - 按模块筛选（保留 AAA、BBB、CCC）
输出到 all_data/data_abc_{sample_ratio}/，数据量更小，训练更快
"""
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

# 复用 convert_oulad 的核心函数
from convert_oulad import (
    load_all_oulad_data,
    build_node_mapping,
    get_student_course_labels,
    compute_course_progress,
    build_edges,
    extract_student_assessment_features,
    extract_student_registration_features,
    extract_student_vle_features,
    extract_course_assessment_features,
    extract_course_vle_features,
    compute_course_statistics,
    build_node_features,
    save_contraTGT_format,
    save_edge_features_and_pairs,
)


def filter_by_modules(student_vle, student_info, courses, assessments,
                      student_assessment, student_registration, vle, modules):
    """按模块筛选所有数据"""
    modules_set = set(m.strip().upper() for m in modules)
    
    student_vle = student_vle[student_vle['code_module'].isin(modules_set)].copy()
    student_info = student_info[student_info['code_module'].isin(modules_set)].copy()
    courses = courses[courses['code_module'].isin(modules_set)].copy()
    assessments = assessments[assessments['code_module'].isin(modules_set)].copy()
    student_registration = student_registration[student_registration['code_module'].isin(modules_set)].copy()
    vle = vle[vle['code_module'].isin(modules_set)].copy()
    
    # student_assessment 通过 assessments 关联筛选
    kept_assessment_ids = set(assessments['id_assessment'].unique())
    student_assessment = student_assessment[student_assessment['id_assessment'].isin(kept_assessment_ids)].copy()
    
    return (student_vle, student_info, courses, assessments,
            student_assessment, student_registration, vle)


def main():
    import argparse
    
    print("=" * 80)
    print("OULAD 数据转换 - 按模块筛选（AAA、BBB、CCC）")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description='OULAD 数据转换 - 按模块筛选')
    parser.add_argument('--modules', type=str, default='AAA,BBB,CCC',
                        help='保留的模块，逗号分隔，如 AAA,BBB,CCC')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='边数据采样比例 (0.0-1.0)')
    parser.add_argument('--max_edges', type=int, default=None,
                        help='最大边数限制')
    args = parser.parse_args()
    
    modules = [m.strip() for m in args.modules.split(',') if m.strip()]
    if not modules:
        print("错误: 请指定至少一个模块，如 --modules AAA,BBB,CCC")
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    oulad_data_dir = os.path.join(script_dir, 'OULAD-main', 'data')
    
    # 输出目录: data_abc_{sample_ratio}
    if args.sample_ratio == 1.0:
        sample_ratio_str = '1'
    else:
        sample_ratio_str = str(args.sample_ratio)
    output_dir = os.path.join(script_dir, 'all_data', f'data_abc_{sample_ratio_str}')
    node_feature_dir = output_dir
    
    if not os.path.exists(oulad_data_dir):
        print(f"错误: 找不到数据目录 {oulad_data_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    pbar = tqdm(total=8, desc="转换进度", unit="步")
    
    # 加载数据
    pbar.set_postfix_str("加载数据")
    (student_vle, student_info, courses, assessments,
     student_assessment, student_registration, vle) = load_all_oulad_data(oulad_data_dir)
    print(f"  原始 studentVle: {len(student_vle):,} 行")
    print(f"  原始 studentInfo: {len(student_info):,} 行")
    pbar.update(1)
    
    # 按模块筛选
    pbar.set_postfix_str("按模块筛选")
    print(f"\n按模块筛选: {', '.join(modules)}")
    (student_vle, student_info, courses, assessments,
     student_assessment, student_registration, vle) = filter_by_modules(
        student_vle, student_info, courses, assessments,
        student_assessment, student_registration, vle, modules
    )
    print(f"  筛选后 studentVle: {len(student_vle):,} 行")
    print(f"  课程数: {len(courses)}")
    pbar.update(1)
    
    # 构建节点映射（基于筛选后的 student_vle）
    pbar.set_postfix_str("节点映射")
    student_to_id, course_to_id, num_students, num_courses = build_node_mapping(student_info, student_vle)
    print(f"  学生节点数: {num_students:,}")
    print(f"  课程节点数: {num_courses}")
    pbar.update(1)
    
    # 提取特征
    pbar.set_postfix_str("提取特征")
    student_ids = list(student_to_id.keys())
    student_assessment_features = extract_student_assessment_features(
        student_assessment, assessments, student_ids
    )
    student_registration_features = extract_student_registration_features(
        student_registration, student_ids
    )
    student_vle_features = extract_student_vle_features(student_vle, student_ids)
    
    course_keys = list(course_to_id.keys())
    course_assessment_features, _ = extract_course_assessment_features(assessments, course_keys)
    course_vle_features = extract_course_vle_features(vle, course_keys)
    course_stats = compute_course_statistics(student_vle, courses, course_to_id)
    pbar.update(1)
    
    # 课程进度特征 + 构建边
    pbar.set_postfix_str("课程进度与边")
    course_dates, participation_first_third = compute_course_progress(student_vle, courses)
    label_map = get_student_course_labels(student_info)
    edges_df = build_edges(
        student_vle, student_to_id, course_to_id, label_map,
        course_dates=course_dates, participation_first_third=participation_first_third,
    )
    print(f"  初始边数: {len(edges_df):,}")
    pbar.update(1)
    
    # 采样
    original_edge_count = len(edges_df)
    if args.sample_ratio < 1.0:
        edges_df = edges_df.sort_values('ts').reset_index(drop=True)
        sample_size = int(len(edges_df) * args.sample_ratio)
        edges_df = edges_df.head(sample_size).reset_index(drop=True)
        print(f"  采样后: {original_edge_count:,} -> {len(edges_df):,} 条边 ({args.sample_ratio*100:.1f}%)")
    
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
    pbar.update(1)
    
    # 构建节点特征
    pbar.set_postfix_str("节点特征")
    used_student_ids = set(student_to_id.keys())
    student_info_filtered = student_info[student_info['id_student'].isin(used_student_ids)].copy()
    node_features, feature_dim = build_node_features(
        student_info_filtered, num_students, num_courses, course_to_id,
        course_stats, student_assessment_features, student_registration_features,
        student_vle_features, course_assessment_features, course_vle_features
    )
    pbar.update(1)
    
    # 保存
    pbar.set_postfix_str("保存")
    pbar.update(1)
    edges_path, feature_path = save_contraTGT_format(
        edges_df, node_features, output_dir,
        student_to_id, course_to_id, node_feature_dir
    )
    edge_feat_path, pair_path = save_edge_features_and_pairs(edges_df, output_dir)
    if edge_feat_path:
        print(f"边特征(课程进度): {edge_feat_path}")
    if pair_path:
        print(f"学生-课程对(去重用): {pair_path}")
    
    pbar.close()
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"边数据: {edges_path}")
    print(f"节点特征: {feature_path}")
    print(f"总边数: {len(edges_df):,}")
    print(f"节点数: {num_students + num_courses:,} (学生: {num_students:,}, 课程: {num_courses})")
    print(f"特征维度: {feature_dim}")
    print(f"\n使用方式: --data_dir data_abc_{sample_ratio_str}")


if __name__ == '__main__':
    main()
