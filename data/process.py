import json

# 文件路径列表
file_paths = [
    # retriever/radiology
    '/ai/xxr/LLM/MMed-RAG/data/training/retriever/radiology/radiology_train.json',
    '/ai/xxr/LLM/MMed-RAG/data/training/retriever/radiology/radiology_val.json',

    # alignment/radiology
    '/ai/xxr/LLM/MMed-RAG/data/training/alignment/radiology/radiology_report.json',
    '/ai/xxr/LLM/MMed-RAG/data/training/alignment/radiology/radiology_vqa.json',

    # retriever/pathology
    '/ai/xxr/LLM/MMed-RAG/data/training/retriever/pathology/pathology_train.json',
    '/ai/xxr/LLM/MMed-RAG/data/training/retriever/pathology/pathology_val.json',

    # alignment/pathology
    '/ai/xxr/LLM/MMed-RAG/data/training/alignment/pathology/pathology_vqa.json'
]

for file_path in file_paths:
    # 读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 更新 "image_root" 值
    data = [item for item in data if '/home/wenhao/Datasets/med/rad/mimic-cxr-jpg' not in item.get('image_root', '')]
    for item in data:
        if 'image_root' in item and 'iu_xray' in item['image_root']:
            item['image_root'] = '/ai/xxr/Datasets/Med/iu_xray/images'
        elif 'image_root' in item and 'pmc_oa' in item['image_root']:
            item['image_root'] = '/ai/xxr/Datasets/Med/pmc_oa/images'
        elif 'image_root' in item and 'quilt_1m' in item['image_root']:
            item['image_root'] = '/ai/xxr/Datasets/Med/quilt_1m'
        # # 更新 "split" 值
        # if 'split' in item:
        #     item['split'] = 'val'

    # 将更新后的数据写回 JSON 文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"文件 {file_path} 的 'image_root' 值已更新。")