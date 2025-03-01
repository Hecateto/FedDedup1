from ep_mpd import MultiPartyDeduplicator, EgPsiType, EgPsiDataType, create_int_elements_pairwise
import random
import argparse

parser = argparse.ArgumentParser(description="Runs the EP-MPD deduplication protocol")
parser.add_argument('--psi-type', type=int, help="EG-PSI Type (1 or 2). Default is 1.", default=1)
parser.add_argument('--num-clients', type=int, help="Number of clients (Integer). Default is 10.", default=10)
parser.add_argument('--num-ele', type=int, help="Number of elements in each client's dataset (Integer). Default is 10.", default=10)
parser.add_argument('--seed', type=int, help="Random seed for dataset creation (Integer). Default is 42.", default=42)
parser.add_argument('--dup-per', type=float, help="Percentage of duplicates (Between 0.0 and 1.0). Default is 0.3.", default=0.3)

args = parser.parse_args()

eg_psi_type = EgPsiType.TYPE1 if args.psi_type == 1 else EgPsiType.TYPE2
num_elements = args.num_ele
num_clients = args.num_clients
dup_per = args.dup_per

random.seed(args.seed)

client_data = []

# 创建包含重复数据的客户端数据
client_data_dict = create_int_elements_pairwise(num_clients, num_elements, dup_per)

# 累加，形成未去重的全集
for i in client_data_dict:
    client_data.append(client_data_dict[i])
non_duplicated_list = []

for data in client_data:
    non_duplicated_list.extend(data)

# 去重
mpd = MultiPartyDeduplicator(client_data=client_data, data_type=EgPsiDataType.INT, eg_type=eg_psi_type)
mpd.deduplicate()
mpd_full_dataset = mpd.get_combined_dataset()

# 检查去重后的数据是否正确
client_data_full = []
for data in client_data:
    client_data_full += data
client_data_full = list(set(client_data_full))  # 去重
client_data_full.sort()

mpd_full_dataset.sort()

for x, y in zip(client_data_full, mpd_full_dataset):
    assert x == y

print("EG PSI Type: ", eg_psi_type)

mpd.print_timing_stats()

print("\nTotal data with duplicates: ", len(non_duplicated_list))
print("Total data without duplicates: ", len(mpd_full_dataset))
