import json
import sys


def get_img_map(img_map_file):
    img_map_res = {}
    for idx, img_sku in enumerate(open(img_map_file).readlines()):
        img_sku = img_sku.split(".")[0]
        img_map_res[img_sku] = idx
    return img_map_res


def mender_data(in_json_file, out_dir, name_type):

    f_sku = open(out_dir + '/' + name_type + '.sku', 'w')
    f_src = open(out_dir + '/' + name_type + '.src', 'w')
    f_tgt = open(out_dir + '/' + name_type + '.tgt', 'w')

    # for sku, item in json.load(open(in_json_file)).items():
    for cur_content in json.load(open(in_json_file)):
        # sku = list(cur_content.keys())[0]
        # item = cur_content[sku]
        # if item['tgt']:
        #     source = "".join(item['src'].strip().split())
        #     for target in item['tgt']:
        #         target = "".join(target.strip().split())
        #         f_sku.write(sku + '\n')
        #         f_src.write(source + '\n')
        #         f_tgt.write(target + '\n')
        # else:
        #     sys.stderr.write("Error Sku: {} \n".format(sku))

        sku = cur_content['idx']
        source = "".join(cur_content['src'].strip().split())
        items = cur_content['tgt']
        if items:
            for target in items:
                target = "".join(target.strip().split())
                f_sku.write(sku + '\n')
                f_src.write(source + '\n')
                f_tgt.write(target + '\n')
        else:
            sys.stderr.write("Error Sku: {} \n".format(sku))

    f_sku.close()
    f_src.close()
    f_tgt.close()

    sys.stdout.write("The Task Has Been Finished!\n")


if __name__ == '__main__':
    in_json_file = sys.argv[1]
    out_dir = sys.argv[2]
    name_type = sys.argv[3]
    mender_data(in_json_file, out_dir, name_type)
