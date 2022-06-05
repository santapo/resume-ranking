import os
import glob
import shutil

all_cv_files = glob.glob(os.path.join("Data/CV/New data", "*.pdf"))
all_cv_files += glob.glob(
    os.path.join("/Users/santapo/Works/Techainer/yody-hackathon/Data/CV_Loại tốt nghiệp/CV_Graduation", "*", "*.pdf"))
all_cv_files += glob.glob(
    os.path.join("/Users/santapo/Works/Techainer/yody-hackathon/Data/CV_Loại tốt nghiệp/CV_Graduation", "*", "*.doc"))
all_cv_files += glob.glob(
    os.path.join("/Users/santapo/Works/Techainer/yody-hackathon/Data/CV_Loại tốt nghiệp/CV_Graduation", "*", "*.docx"))

for idx, cv_file in enumerate(all_cv_files):
    print(idx, cv_file)
    file_extension = os.path.splitext(cv_file)[1]
    new_cv_file = os.path.join("Data/revamped", "cv_" + str(idx) + file_extension)
    shutil.copy(cv_file, new_cv_file)