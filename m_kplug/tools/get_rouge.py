import files2rouge
import os
import sys
from tqdm import tqdm, trange

cur_path = os.path.dirname(os.path.abspath(__file__))
base_path = cur_path
sys.path.insert(0, base_path)
from file_util import delete_file

ref = sys.argv[1]
result = sys.argv[2]

ref_list = open(ref).readlines()
result_list = open(result).readlines()

f_ref_path = ref + 'ref_file'
f_result_path = result + 'hyp_file'

f_ref = open(f_ref_path, 'w')
f_result = open(f_result_path, 'w')

for i in trange(len(ref_list)):
    ref = ''.join(ref_list[i].strip().split())
    result = ''.join(result_list[i].strip().split())

    if not ref or not result:
        continue
    w2id = {}
    idx = 0
    for w in ref + result:
        if w not in w2id:
            w2id[w] = str(idx)
            idx += 1
    ref_ids = [w2id[w] for w in ref]
    result_ids = [w2id[w] for w in result]
    f_ref.write(' '.join(ref_ids) + '\n')
    f_result.write(' '.join(result_ids) + '\n')

f_ref.close()
f_result.close()

# files2rouge.run('hyp_file', 'ref_file')

files2rouge.run(f_result_path, f_ref_path)

delete_file(f_result_path)
delete_file(f_ref_path)