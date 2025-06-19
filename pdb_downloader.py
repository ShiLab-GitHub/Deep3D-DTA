import os
import requests

URL = r"https://www.rcsb.org/downloads/"
"""
注：
with open可以创建txt,csv,pdb
requests.get().content可以根据网址下载相关pdb文件
"""


# def download(data_type: str):
#     os.makedirs(f'./{data_type}_pdb')
#     with open(f'.{data_type}_examples.txt', 'r') as f:
#         for line in f:
#             name = line.split('_')[0]
#             download_path = os.path.join(URL, f'{data_type}_pdb')
#             with open(f'./{data_type}_pdb/{name}.pdb', 'wb') as fi:
#                 fi.write(requests.get(download_path).content）
#
#                 download('test)
