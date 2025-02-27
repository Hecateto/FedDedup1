from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from collections import defaultdict

from typing import List, Tuple
from ep_mpd.eg_psi.utils import encode_element
from ep_mpd.eg_psi.type1.prp_key import PRPKey
from ep_mpd.eg_psi.utils import EgPsiDataType


class ClientType1:

    def __init__(self, client_id: int, data_type: EgPsiDataType):
        self.v = defaultdict(list)  # stores the intersection with clients in other group
        self.keys = None  # keys with every client in other group. list of tuples (client_id, key) [客户端id, 密钥]
        self.s = None  # plaintext elements
        self.s_dirty_bits = {}  # key is plaintext elements and value is 0/1 where 0 if element is deleted
        self.s_bytes = {}  # plaintext ele to padded bytes   经过填充后的明文元素
        self.id = client_id
        self.data_type = data_type
        self.group = None
        self.ciphers = {}  # map[密钥->加密器对象]
        self.s_prime = []  # encrypted elements     似乎是多余的，后面代码中没有用到

    def set_group(self, client_group: int):
        self.group = client_group

    def set_keys(self, keys: List[Tuple[int, PRPKey]]):
        self.keys = keys
        for key in self.keys:
            cipher = Cipher(algorithms.AES(key[1].key), modes.CBC(key[1].iv))
            self.ciphers[key[0]] = cipher

    def create_set(self, data_set: List[int]):
        self.s = data_set
        for ele in self.s:
            self.s_dirty_bits[ele] = 1
            ele_bytes = encode_element(ele)
            padder = padding.PKCS7(128).padder()
            ele_bytes = padder.update(ele_bytes) + padder.finalize()
            self.s_bytes[ele] = ele_bytes

    # 将己方的明文元素使用另一组各客户端的密钥进行加密
    # 己方元素数量为m, 另一组客户端数量为n, 则总共有m*n个加密元素
    def encrypt_elements(self, other_client_ids: List[int]):

        for client_id in other_client_ids:
            for idx, ele in enumerate(self.s):
                if self.s_dirty_bits[ele] == 1:
                    ele_bytes = self.s_bytes[ele]

                    encryptor = self.ciphers[client_id].encryptor()
                    enc_ele = encryptor.update(ele_bytes) + encryptor.finalize()

                    self.s_prime.append((idx, enc_ele)) # [元素索引, 加密元素]
                    # 这里同一个元素索引对应的加密元素有多个

    def set_intersection(self, r: List[bytes]):

        for enc_idx in r:
            ele = self.s[enc_idx]
            if self.group == 0:
                if self.s_dirty_bits[ele] == 1:
                    self.s_dirty_bits[ele] = 0      # 直接删除

        self.reset_client()

    def get_deduplicated_dataset(self):
        new_s = []
        for ele in self.s_dirty_bits:
            if self.s_dirty_bits[ele] == 1:
                new_s.append(ele)           # 根据脏位判断是否删除

        return new_s

    def reset_client(self):
        """
        We recursively create groups up the binary tree, so reset everything except the updated client set
        :return:
        """
        self.group = None
        self.s_prime.clear()  # encrypted elements

    def __str__(self) -> str:
        return "Client ID: {}, Client Group: {}".format(self.id, self.group)
