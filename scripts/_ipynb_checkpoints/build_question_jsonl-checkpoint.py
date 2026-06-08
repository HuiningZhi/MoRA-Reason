import os
import json


if __name__ == "__main__":
    file_path = "LLaVA_data_all/3vqa/test_slake.json"

    with open(file_path, "r") as f:
        data = json.load(f)

    save_file_path = "LLaVA_data_all/3vqa/test_slake.jsonl"
    ans_file = open(save_file_path, "w")
    new_data = []
    for sample in data:
        temp = {}
        temp["question_id"] = sample["id"]
        temp["image"] = sample["image"]
        temp["text"] = sample["conversations"][0]["value"]
        temp["answer_type"] = sample["answer_type"]
        ans_file.write(
            json.dumps(
                temp
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()