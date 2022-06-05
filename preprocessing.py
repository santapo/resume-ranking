"""
TODO: + Field Normalization
      + 

"""
from datetime import datetime, date
import re
import nltk
from nltk.corpus import stopwords


uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
 
 
def loaddicchar():
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic
 
 
dicchar = loaddicchar()
 
# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def convert_unicode(txt):
    return re.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def validate_phone(phone: str):
    validate_phone_pattern = "^(03|05|07|08|09|01[2|6|8|9])([0-9]{8})$"
    if re.match(validate_phone_pattern, phone):
        return phone
    return ""     # "Can't Extracted"

def validate_name():
    ...

def norm_name(name):
    return name.title()

def norm_university():
    ...

def norm_company():
    ...

def norm_birthdate(text):
    for fmt in ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return ""

def diff_month(d1, d2):
    d1 = datetime.strptime(d1, "%m/%Y")
    d2 = datetime.strptime(d2, "%m/%Y")
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def age(birthdate):
    today = date.today()
    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    return age

# def norm_skills(skills):

def validate_gender(gender):
    lower_gender = gender.lower()
    valid_gender = ["male", "female", "nam", "nữ", "nu"]
    if lower_gender not in valid_gender:
        return ""
    return gender

def norm_gender(gender):
    gender = gender.lower()
    norm_dict = {
        "male": ["nam", "male"],
        "female": ["female", "nữ", "nu"]
    }
    for k, v in norm_dict.items():
        if gender in v:
            return k
    return ""

def text_preprocess(text):
    text = convert_unicode(text)
    text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
    tokens = nltk.word_tokenize(text)
    vie_stop_words = set(stopwords.words('vietnamese'))
    en_stop_words = set(stopwords.words('english'))

    words = [w for w in tokens if (w not in vie_stop_words) or (w not in en_stop_words)]
    text = ' '.join(words)

    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def preprocessing(json_file: dict):
    json_file["info_phone"] = validate_phone(json_file["info_phone"])
    try:
        json_file["info_birthdate"] = norm_birthdate(json_file["info_birthdate"])
        json_file["info_age"] = ""
        if json_file["info_birthdate"]:
            json_file["info_age"] = age(json_file["info_birthdate"])
        json_file["info_birthdate"] = json_file["info_birthdate"].strftime("%d/%m/%Y")
    except:
        pass
    json_file["info_gender"] = norm_gender(validate_gender(json_file["info_gender"]))
    json_file["info_name"] = norm_name(json_file["info_name"])

    for idx, exp in enumerate(json_file["experience"]):
        try:
            start_time = json_file["experience"][idx]["normalized_period"]["start"]
            end_time = json_file["experience"][idx]["normalized_period"]["end"]
            if end_time == "":
                end_time = datetime.now().strftime("%m/%Y")
            json_file["experience"][idx]["total_time"] = diff_month(end_time, start_time)

            json_file["experience"][idx]["exp_description"] = text_preprocess(json_file["experience"][idx]["exp_description"])
        except:
            json_file["experience"][idx]["total_time"] = ""

    json_file["others_skills"] = " ".join([json_file["others_skills"], json_file["others_certificate"]])
    json_file["others_skills"] = text_preprocess(json_file["others_skills"])
    
    return json_file

if __name__ == "__main__":
    import json
    json_file = json.load(open("CV_3_2022_en.json"))
    json_file = preprocessing(json_file)
    json.dump(json_file, open("cv_14_2.json", "w"), indent=4)