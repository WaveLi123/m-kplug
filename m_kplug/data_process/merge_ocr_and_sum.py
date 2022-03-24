"""
合并OCR数据和摘要数据。
"""

def get_title_and_content_from_ocr():

    f_input = '/workspace/fairseq/data-master/data-nlp/JD/discovery_meta/raw/discovery_all.ocr'
    f_out_title = open(f_input + '.title', 'w')
    f_out_content = open(f_input + '.content', 'w')

    for i, line in enumerate(open(f_input)):
        ocr_data = line.strip().split('[SEP]')
        title = ocr_data[0]
        content = '[SEP]'.join(ocr_data[1:])[:500]
        content_len = len(''.join(ocr_data[1:]).strip())
        if len(title) > 3 and content_len > 8:
            f_out_title.write(title + '\n')
            f_out_content.write(content + '\n')
        if i > 2000000:
            break

get_title_and_content_from_ocr()