from collections import Counter
import json

# def sample_counter(dataloader):
#     class_counts = Counter()

#     for _, label, _ in dataloader:
#         print(label)
#         class_counts[label] += 1

#     for class_id, count in class_counts.items():
#         print(f"Class {class_id}: {count} samples")

#     class_counts_dict = dict(class_counts)
#     with open('class_counts.txt', 'w') as file:
#         file.write(json.dumps(class_counts_dict, indent=4))

def sample_counter(dataloader):
    class_counts = Counter()

    for _, batch_labels, _ in dataloader:
        print(batch_labels)
        class_counts.update(batch_labels.tolist())
    
    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples")

    class_counts_dict = dict(class_counts)
    with open('data_txt/ImageNet_LT/imagenetlt_class_count.txt', 'w') as file:
        file.write(json.dumps(class_counts_dict, indent=4))