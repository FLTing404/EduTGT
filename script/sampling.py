from collections import defaultdict
import bisect
import numpy as np



#spatial sampling
def get_adj_list(edge):
    adj_list = defaultdict(list)
    for i,edg in enumerate(edge['idx']):
        st = int(edg[0])
        en = int(edg[1])
        ts = int(edg[2])
        idx = int(edg[3])
        adj_list[st].append((en,ts,idx))
        adj_list[en].append((st,ts,idx)) 
    return adj_list

def init_offset(adj_list):
    node_l = []
    ts_l = []
    idx_l = []
    offset_l = [0]
    for i in range(len(adj_list)):
        curr = adj_list[i]  
        curr = sorted(curr, key=lambda x: x[2])  
        node_l.extend([x[0] for x in curr])  
        ts_l.extend([x[1] for x in curr])  
        idx_l.extend([x[2] for x in curr])  
        offset_l.append(len(node_l))  

    node_l = np.array(node_l)
    ts_l = np.array(ts_l)
    idx_l = np.array(idx_l)
    offset_l = np.array(offset_l)

    return node_l, ts_l, idx_l, offset_l

def find_before(node_l, ts_l, idx_l, offset_l, node, ts):
    ngh_node_l = node_l[offset_l[node]:offset_l[node + 1]]  
    ngh_ts_l = ts_l[offset_l[node]:offset_l[node + 1]]  
    ngh_idx_l = idx_l[offset_l[node]:offset_l[node + 1]]  

    if len(ngh_node_l) == 0 or len(ngh_ts_l) == 0:
        return ngh_node_l, ngh_ts_l, ngh_idx_l

    left = 0
    right = len(ngh_node_l) - 1
    while left + 1 < right:  
        mid = (left + right) // 2
        curr_t = ngh_ts_l[mid]
        if curr_t < ts:
            left = mid
        else:
            right = mid

    if ngh_ts_l[right] < ts:  
        return ngh_node_l[:right], ngh_idx_l[:right], ngh_ts_l[:right]
    else:
        return ngh_node_l[:left], ngh_idx_l[:left], ngh_ts_l[:left]


def get_neighbor_list(node_l, ts_l, idx_l, offset_l, node, timestamp, num_sample):
    #     print(timestamp)
    ngh_node = np.zeros((len(node), num_sample)).astype(np.int32)  
    ngh_ts = np.zeros((len(node), num_sample)).astype(np.int32)  
    ngh_idx = np.zeros((len(node), num_sample)).astype(np.int32)  
    ngh_mask = np.zeros((len(node), num_sample)).astype(np.int32)  
    num_sample = num_sample - 1
    for i, edg in enumerate(zip(node, timestamp)):
        node, idx, ts = find_before(node_l, ts_l, idx_l, offset_l, edg[0], edg[1])  
        node = node[::-1]
        idx = idx[::-1]
        ts = ts[::-1]
        #         print(ts)
        #         print('*********************')
        #         print(int(edg[1])-ts)
        if len(node) > 0:
            if len(node) > num_sample:  
                node = node[:num_sample]
                idx = idx[:num_sample]
                ts = ts[:num_sample]
            ngh_node[i, 1:len(node) + 1] = node
            ngh_ts[i, 1:len(node) + 1] = int(edg[1]) - ts
            ngh_idx[i, 1:len(node) + 1] = idx
            ngh_mask[i, 1:len(node) + 1] = 1

        ngh_node[i, 0] = edg[0]
        ngh_mask[i, 0] = 1
    return ngh_node, ngh_ts, ngh_idx, ngh_mask

#temporal sampling
def get_interaction_list(edge):
    """每个节点对应其参与边的 idx 列表（按边顺序），并返回排序后的副本供二分查找。"""
    idx_list = defaultdict(list)
    for i, edg in enumerate(edge['idx']):
        st = int(edg[0])
        en = int(edg[1])
        idx = int(edg[3])
        idx_list[st].append(idx)
        idx_list[en].append(idx)
    # 每个节点保留按 idx 值排序的数组，用于 O(log degree) 二分查找
    idx_list_sorted = {}
    for n, lst in idx_list.items():
        idx_list_sorted[n] = np.sort(np.array(lst, dtype=np.int64)) if lst else np.array([], dtype=np.int64)
    return idx_list, idx_list_sorted


def _find_index_in_sorted(sorted_arr, idx):
    """在已排序数组中找 idx 的位置：若存在则返回其下标，否则返回比 idx 小的最大元素下标。"""
    if len(sorted_arr) == 0:
        return -1
    pos = bisect.bisect_left(sorted_arr, idx)
    if pos < len(sorted_arr) and sorted_arr[pos] == idx:
        return pos
    return pos - 1


def _edges_slice_to_node_time(edge_slice, ts, reverse=True):
    """将边切片 (N, 4) 转为 node/time 向量：node[2j]=v, node[2j+1]=u, time 为 ts - edge_ts。"""
    n_ed = edge_slice.shape[0]
    if n_ed == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    node = np.empty(2 * n_ed, dtype=np.int32)
    node[0::2] = edge_slice[:, 1]
    node[1::2] = edge_slice[:, 0]
    time = np.repeat(ts - edge_slice[:, 2].astype(np.int64), 2).astype(np.int32)
    if reverse:
        node = node[::-1].copy()
        time = time[::-1].copy()
    return node, time


def get_unique_node_sequence(b_edge, f_edge, k, idx_list, flag, idx_list_sorted=None):
    """
    优化版：用 idx_list_sorted 二分查找替代 .index()，用预取边数组 + 向量化切片替代逐条 enumerate。
    get_interaction_list 现在返回 (idx_list, idx_list_sorted)，调用方需传入 idx_list_sorted；
    若未传则退化为仅用 idx_list 的二分查找（需保证 idx_list 按值有序）。
    """
    num_nodes = len(b_edge['idx'])
    node_sequence = np.zeros((num_nodes, k), dtype=np.int32)
    node_timestamp = np.zeros((num_nodes, k), dtype=np.int32)
    node_seq_mask = np.zeros((num_nodes, k), dtype=np.int32)
    row = int(k / 2)

    # 预取全边数组，避免在循环里重复索引；列顺序: FromNodeId, ToNodeId, TimeStep, idx
    if hasattr(f_edge['idx'], 'numpy'):
        f_edge_np = f_edge['idx'].numpy()
    else:
        f_edge_np = np.asarray(f_edge['idx'], dtype=np.int64)

    n_edges = f_edge_np.shape[0]
    if idx_list_sorted is None:
        idx_list_sorted = {n: np.array(lst) if lst else np.array([], dtype=np.int64) for n, lst in idx_list.items()}

    for i in range(num_nodes):
        edg = b_edge['idx'][i]
        if hasattr(edg, 'numpy'):
            edg = edg.numpy()
        else:
            edg = np.asarray(edg)
        ts = int(edg[2])
        idx = int(edg[3])
        from_node = int(edg[0]) if flag else int(edg[1])

        sorted_arr = idx_list_sorted.get(from_node)
        if sorted_arr is None or len(sorted_arr) == 0:
            node_timestamp[i, 0] = 0
            node_sequence[i, 0] = from_node
            node_seq_mask[i, 0] = 1
            continue

        index = _find_index_in_sorted(sorted_arr, idx)
        if index <= 0:
            node_timestamp[i, 0] = 0
            node_sequence[i, 0] = from_node
            node_seq_mask[i, 0] = 1
            continue

        last_event = int(sorted_arr[index - 1])
        last_event = min(last_event, n_edges)

        if last_event < row:
            edge_slice = f_edge_np[0:last_event]
            node, time = _edges_slice_to_node_time(edge_slice, ts, reverse=True)
            # 修复：确保 node 和 time 长度一致，避免广播错误
            L = min(len(node), len(time), k - 1)
            if L > 0:
                node_timestamp[i, 1:1 + L] = time[:L]
                node_sequence[i, 1:1 + L] = node[:L]
                node_seq_mask[i, 1:1 + L] = 1
        else:
            start = max(0, last_event - row)
            edge_slice = f_edge_np[start:last_event]
            node, time = _edges_slice_to_node_time(edge_slice, ts, reverse=True)
            # 修复：确保 node 和 time 长度一致，避免广播错误
            L = min(len(node), len(time), k - 1)
            if L > 0:
                node_timestamp[i, 1:1 + L] = time[:L]
                node_sequence[i, 1:1 + L] = node[:L]
                node_seq_mask[i, 1:1 + L] = 1

        node_timestamp[i, 0] = 0
        node_sequence[i, 0] = from_node
        node_seq_mask[i, 0] = 1

    return node_sequence, node_seq_mask, node_timestamp
