import csv
import random
import os

input_file = "UTTRANSINFO.txt"
output_dir = "Data/processed/output"
train_ratio = 0.8
dev_ratio = 0.1
random.seed(42)


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mandarin = row["PROMPT"].strip()
            wu = row["TRANSCRIPTION"].strip()
            if mandarin and wu:
                data.append((mandarin, wu))
    return data


def save_csv(data, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mandarin", "wu"])
        for m, w in data:
            writer.writerow([m, w])


def split_data(data):
    random.shuffle(data)
    n = len(data)
    
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train = data[:n_train]
    dev = data[n_train : n_train + n_dev]
    test = data[n_train + n_dev :]

    return train, dev, test


if __name__ == "__main__":
    data = load_data(input_file)
    print(f"Total pairs: {len(data)}")

    train, dev, test = split_data(data)

    os.makedirs(output_dir, exist_ok=True)

    save_csv(train, f"{output_dir}/train.csv")
    save_csv(dev, f"{output_dir}/dev.csv")
    save_csv(test, f"{output_dir}/test.csv")

    print(f"train = {len(train)}, dev = {len(dev)}, test = {len(test)}")