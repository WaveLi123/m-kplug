import json
import sys
import random


def mender_data(in_json_file, out_path, split_num):
    res = json.load(open(in_json_file))
    new_res = []
    res_keys = list(res.keys())
    random.shuffle(res_keys)

    for key in res_keys:
        new_res.append({key: res[key]})

    sys.stdout.write("Mender Json Data Starts. All Size is {}\n ".format(len(new_res)))
    if split_num > 0:
        size = int(len(new_res) / split_num)
        for num in range(split_num + 1):
            if (num + 1) * size < len(new_res):
                part_res = new_res[num * size: (num + 1) * size]
            else:
                part_res = new_res[num * size:]
            with open(out_path + '/train_part_' + str(num) + '.json', 'w') as f_w:
                if len(part_res) > 0:
                    res = json.dumps(part_res, ensure_ascii=False, indent=2)
                    f_w.write(res)
            sys.stdout.write("Mender Part {} Json Data Ends. Size is {}\n ".format(num, len(part_res)))
    else:
        with open(out_path, 'w') as f_w:
            res = json.dumps(new_res, ensure_ascii=False, indent=2)
            f_w.write(res)

        sys.stdout.write("Mender Json Data Ends. Size is {}\n ".format(len(new_res)))

    sys.stdout.write("The Task Has Been Finished!\n")


if __name__ == '__main__':
    in_json_file = sys.argv[1]
    out_json_file = sys.argv[2]
    split_num = sys.argv[3]
    mender_data(in_json_file, out_json_file, int(split_num))
