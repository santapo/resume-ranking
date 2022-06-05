import glob
import json
import multiprocessing
import os

import requests
from tqdm import tqdm


def send_single_request(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    response = requests.post(url="techainer.censored.xyzi", files={"file": open(file_path, "rb")})
    json.dump(response.json(),
              open(os.path.join("Data/dumped_json/", "{}.json".format(file_name)), "w"), indent=4)

# all_cv_paths = glob.glob(os.path.join("Data/revamped", "*.pdf"))
# all_cv_paths += glob.glob(os.path.join("Data/revamped", "*.doc"))
# all_cv_paths += glob.glob(os.path.join("Data/revamped", "*.docx"))
all_cv_paths = ["CV_3_2022_en.pdf"]
# pool = multiprocessing.Pool(processes=2)
# output = list(tqdm(
#         pool.imap(send_single_request, all_cv_paths), total=len(all_cv_paths), desc="Requesting...")
#     )
# pool.terminate()
for idx, cv_path in tqdm(enumerate(all_cv_paths)):
    print("\n Requesting {}/{}".format(idx, len(all_cv_paths)))
    file_name = os.path.splitext(os.path.basename(cv_path))[0]
    try:
        response = requests.post(url="techainer.censored.xyzi", files={"file": open(cv_path, "rb")})
        json.dump(response.json(), open(os.path.join(".", "{}.json".format(file_name)), "w"), indent=4)
    except Exception as e:
        print(e)