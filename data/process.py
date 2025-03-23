import json

# 文件路径
file_path = '/ai/xxr/LLM/MMed-RAG/data/training/retriever/radiology/radiology_val.json'

# 读取 JSON 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 更新 "image_root" 值
for item in data:
    if 'image_root' in item and item['image_root'] == '/home/wenhao/Datasets/med/rad/iu_xray/images':
        item['image_root'] = '/ai/xxr/Datasets/Med/rad/iu_xray/images'

# 将更新后的数据写回 JSON 文件
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("所有 'image_root' 值已更新。")