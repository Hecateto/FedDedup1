from collections import defaultdict
from typing import List, Tuple

# 泄露信息
# 1. 交集元素频次
# 2. 交集元素索引
# 3. 交集元素的组和客户端
# 原因：这里求交的过程是完全交由TEE做的

class SemiTrustedThirdPartyType1:

    def __init__(self):
        # self.client_s_primes = defaultdict(list)  # map[组->加密元素列表]
        self.superset_s_primes = defaultdict(int)  # map[加密元素->出现次数] cnt （泄露交集元素频次）

    def receive_from_client(self, client_group: int, client_id: int, client_data: List[Tuple[int, bytes]]):
        for _, ele in client_data:
            self.superset_s_primes[ele] += 1    # cnt++

    # Lazy eval when client requests its vector
    def send_to_client(self, client_group: int, client_id: int, client_data: List[Tuple[int, bytes]]) -> List[bytes]:
        r_idx = []
    # 这里client_data是否需要打乱？否则泄露了索引信息。
    # 元素是加密的，不需要考虑。但是索引信息泄露了交集元素的位置信息。      这里如果中断，第三方直到两个客户端的交集元素索引。
    # 可行的办法是预先置换，然后再由客户端解置换。
        for idx, ele in client_data:
            if self.superset_s_primes[ele] > 1:
                r_idx.append(idx)

        return r_idx

    # Reset the for next EG PSI computation
    def clear_all(self):
        # self.client_s_primes.clear()
        self.superset_s_primes.clear()
