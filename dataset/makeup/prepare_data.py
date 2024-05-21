import json
import os
import time

def savejson(data, path):
    with open(path, "w") as f:
        json.dump(data, f)
    f.close()

def toSec(timeStr):
    t = time.strptime(timeStr, "%H:%M:%S")
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

def captiondata_modify(steps):
    modify_data = {} #{video_name(str): caption_info(dict)}
    for i, step in enumerate(steps[0]):
        tmp_dic = {} #keys = ["duration", "timestamps", "sentences"]
        name = step["video_id"]
        #duration
        tmp_dic["duration"] = step["duration"]
        #timestamps & sentences
        tmp_dic["timestamps"] = []
        tmp_dic["sentences"] = []
        for key in step["step"].keys():
            #sentences
            tmp_dic["sentences"].append(step["step"][key]["caption"])
            #timestamps
            startime = toSec(step['step'][key]["startime"])
            endtime = toSec(step['step'][key]["endtime"])
            tmp_dic["timestamps"].append([startime, endtime])

        modify_data[name] = tmp_dic
    return modify_data

with open("./dataset/makeup/steps.json", "r") as f:
    train_steps = f.readlines()
f.close()
train_steps = [json.loads(x) for x in train_steps]

with open("./dataset/makeup/dev_step.json", "r") as f:
    val_steps = f.readlines()
f.close()
val_steps = [json.loads(x) for x in val_steps]

train_data = captiondata_modify(train_steps)
savejson(train_data, os.path.join("./dataset/makeup/makeup_train.json"))

val_data = captiondata_modify(val_steps)
savejson(val_data, os.path.join("./dataset/makeup/makeup_test.json"))


with open("./dataset/makeup/MTVG_test.json", "r") as f:
    MTVG_test_steps = f.readlines()
f.close()
MTVG_test_steps = [json.loads(x) for x in MTVG_test_steps]

def captiondata_modify_test(steps):
    modify_data = {} #{video_name(str): caption_info(dict)}
    for i, step_t in enumerate(steps[0]):
        tmp_dic = {} #keys = ["duration", "timestamps", "sentences"]
        name = step_t["video_id"]
        #duration
        tmp_dic["duration"] = step_t["video_duration"]
        #timestamps & sentences
        tmp_dic["timestamps"] = [[3,5],]
        tmp_dic["sentences"] = [step_t["caption"],]
        tmp_dic["query_idx"] = step_t["query_idx"]
        # for key in step_t["step"].keys():
        #     # step_t["query_idx"]
        #     # [step_t["caption"],]
        #             # tmp_dic["query_idx"] = 
        #     #sentences
        #     tmp_dic["sentences"].append(step_t["step"][key]["caption"])
        #     #timestamps
        #     startime = toSec(step_t['step'][key]["startime"])
        #     endtime = toSec(step_t['step'][key]["endtime"])
        #     tmp_dic["timestamps"].append([startime, endtime])

        modify_data[name+tmp_dic["query_idx"]] = tmp_dic
    return modify_data

MTVG_test_data = captiondata_modify_test(MTVG_test_steps)

savejson(MTVG_test_data, os.path.join("./dataset/makeup/makeup_MTVG_test.json"))
