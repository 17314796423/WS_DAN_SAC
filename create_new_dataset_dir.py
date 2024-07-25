import os
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import random, json

def list_files_in_directory(directory_path):
    # 获取目录下的所有文件名，并存储到一个列表中
    file_list = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return file_list


directory_path = "E:\\考研\\算法分析设计\\2024算法设计与分析-作业汇总\\基于视频序列的鸟类识别\\2022鸟图\\2022鸟图"  # 替换为你的目录路径
files = list_files_in_directory(directory_path)
bird_map = {
    '斑鱼狗': '081.Pied_Kingfisher',
    '家燕': '136.Barn_Swallow',
    '丝光椋鸟': '201.Silk_Glare_Lang',
    '乌鸫': '202.Dark_Thrush',
    '八哥': '203.Eight_Geese',
    '喜鹊': '204.Magpie',
    '山斑鸠': '205.Mountain_Spotted_Dove',
    '戴胜': '206.Great_Tit',
    '棕背伯劳': '207.Brown_Back_Shrike',
    '池鹭': '208.Pond_Heron',
    '灰喜鹊': '209.Gray_Magpie',
    '灰头麦鸡': '210.Gray_Headed_Prairie_Chicken',
    '灰椋鸟': '211.Gray_Lang',
    '牛背鹭': '212.Cattle_Back_Heron',
    '珠颈斑鸠': '213.Pearl_Neck_Spotted_Dove',
    '白头鹎': '214.White_Headed_Bearded',
    '白胸苦恶鸟': '215.White_Breasted_Misery_Bird',
    '白腰草鹬': '216.White_Waist_Sandpiper',
    '白鹡鸰': '217.White_Lark',
    '白鹭': '218.Egret',
    '红尾伯劳': '219.Red_Tailed_Shrike',
    '红隼': '220.Red_Falcon',
    '纯色山鹪莺': '221.Solid_Color_Hill_Warbler',
    '远东山雀': '222.Far_Eastern_Sparrow',
    '雉鸡': '223.Pheasant',
    '鹊鸲': '224.Magpie_Thrush',
    '黄苇鳽': '225.Yellow_Bittern',
    '黑卷尾': '226.Black_Tailed_Flycatcher',
    '黑尾蜡嘴雀': '227.Black_Tailed_Wax_Beak',
    '黑水鸡': '228.Black_Waterfowl',
    '家鸽': '229.Domestic_Pigeon',
    '小鷿鷈': '230.Little_Grebe',
    '普通翠鸟': '231.Common_Kingfisher',
    '金腰燕': '232.Red-rumped_Swallow'
}


def search_and_download_images(keyword, download_dir, txt_dir, num_images=200):
    # 创建保存图片的目录
    os.makedirs(download_dir, exist_ok=True)
    test_path = os.path.join(txt_dir, 'bird_test.txt')
    train_path = os.path.join(txt_dir, 'bird_train.txt')
    all_items = os.listdir(download_dir)
    # 过滤出所有文件
    files = [item for item in all_items if os.path.isfile(os.path.join(download_dir, item))]
    if len(files) > 0:
        return
    rn = 30
    image_urls = []
    while rn <= 210:
        cookies = {
            'winWH': '%5E6_2560x1431',
            'BDIMGISLOGIN': '0',
            'BDqhfp': '%E4%B8%9D%E5%85%89%E6%A4%8B%E9%B8%9F%26%260-10undefined%26%260%26%261',
            'cleanHistoryStatus': '0',
            'MCITY': '-315%3A',
            'BIDUPSID': 'E3B4335AD1F0E6C9E9CA698B56909CBD',
            'PSTM': '1715927349',
            'BDUSS': 'EF2YlRaMlozTGdTbVlzamNQVn5IZkNybVF5Mzl6N24zc3Q5ZFhkUmpwT2txWjltSVFBQUFBJCQAAAAAAAAAAAEAAADqvdA6MTM5MTVnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKQceGakHHhmS',
            'BDUSS_BFESS': 'EF2YlRaMlozTGdTbVlzamNQVn5IZkNybVF5Mzl6N24zc3Q5ZFhkUmpwT2txWjltSVFBQUFBJCQAAAAAAAAAAAEAAADqvdA6MTM5MTVnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKQceGakHHhmS',
            'BDSFRCVID': 'zcKOJexroG3Kj2Qtq838bQenqYHjxXjTDYrE8HDLnjRzg2_VY-fcEG0PtsBmvLtb6jBvogKKL2OTHm_F_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
            'H_BDCLCKID_SF': 'JnK8_C82tC83fP36qROE-tCshUTMetJXfb6Gsl7F5l8-hxcveJ7d3b0R2G0O0xvJyCOq_K5GbJoxOKQphTbI3tI1XaQGLhRd5GQMQxJN3KJmepC9bT3vQxuwyf8D2-biW2tH2Mbdax7rebCxD6Jab-uyM4vBbtjxB5nnV6625DnJh-PGe6L5e5oyeaKs5-7ybCPX3JjV5PK_Hn7zeP-beM4pbt-qJt7et6ndW-b9MDJSsqnmWtnJyPobXP6nBT5KaarGV4QbQx-5flIGQtbjKb_kQN3T-qLO5bRiL66Rtj6-Dn3oyUkKXp0nhJndKPujtJvD0n6b2qR8qRojXnOxeDu3bG7MJnktfJP8VCKbJID5bP365ITMMt_HbUnXKK62aJ0HKt5vWJ5TMCoLhTrWXh03Wh6UbfQ7bN5yLD0aQ4QDShPC-tP5LlLqQPkD3qbdXKTz0-jx3l02Vb3Ee-t2yUKI04rbJ4RMW23roq7mWn6r_nbzD5KbKt0DKU6raJja32OJ-IIEbnLWeIJ9jjC5Djc-DHttJTn8265KWjrtKRTffjrnhPF324tUXP6-hnjy3b7iBDolbPDWhInP0to_KqFnMl7vaq3Ry6r4VpngBnr8V4cTDln4WDuE-PoxJpOJ-2rLLqR2KDQ8E4QvbURvX5Dg3-7LyM5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIE3-oJqCKBhKPC3J',
            'H_WISE_SIDS_BFESS': '60340_60352_60361',
            'indexPageSugList': '%5B%22%E4%B8%9D%E5%85%89%E6%A4%8B%E9%B8%9F%22%2C%22%E5%86%99%E5%87%BAhelloworld%E7%9A%84%E8%A1%A8%E6%83%85%22%5D',
            'H_WISE_SIDS': '60361',
            'BDSFRCVID_BFESS': 'zcKOJexroG3Kj2Qtq838bQenqYHjxXjTDYrE8HDLnjRzg2_VY-fcEG0PtsBmvLtb6jBvogKKL2OTHm_F_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
            'H_BDCLCKID_SF_BFESS': 'JnK8_C82tC83fP36qROE-tCshUTMetJXfb6Gsl7F5l8-hxcveJ7d3b0R2G0O0xvJyCOq_K5GbJoxOKQphTbI3tI1XaQGLhRd5GQMQxJN3KJmepC9bT3vQxuwyf8D2-biW2tH2Mbdax7rebCxD6Jab-uyM4vBbtjxB5nnV6625DnJh-PGe6L5e5oyeaKs5-7ybCPX3JjV5PK_Hn7zeP-beM4pbt-qJt7et6ndW-b9MDJSsqnmWtnJyPobXP6nBT5KaarGV4QbQx-5flIGQtbjKb_kQN3T-qLO5bRiL66Rtj6-Dn3oyUkKXp0nhJndKPujtJvD0n6b2qR8qRojXnOxeDu3bG7MJnktfJP8VCKbJID5bP365ITMMt_HbUnXKK62aJ0HKt5vWJ5TMCoLhTrWXh03Wh6UbfQ7bN5yLD0aQ4QDShPC-tP5LlLqQPkD3qbdXKTz0-jx3l02Vb3Ee-t2yUKI04rbJ4RMW23roq7mWn6r_nbzD5KbKt0DKU6raJja32OJ-IIEbnLWeIJ9jjC5Djc-DHttJTn8265KWjrtKRTffjrnhPF324tUXP6-hnjy3b7iBDolbPDWhInP0to_KqFnMl7vaq3Ry6r4VpngBnr8V4cTDln4WDuE-PoxJpOJ-2rLLqR2KDQ8E4QvbURvX5Dg3-7LyM5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIE3-oJqCKBhKPC3J',
            'BAIDUID': '58989DE417F26A77CAB8B758EE7097EC:FG=1',
            'BAIDUID_BFESS': '58989DE417F26A77CAB8B758EE7097EC:FG=1',
            'H_PS_PSSID': '60278_60359_60386_60426',
            'BA_HECTOR': '852h2k04a5a5252l24al01852di21e1j8crc01v',
            'ZFY': '4zNmYnZ8XUNbHKn0xqxzeaLINcR8mhO9Jv:BvlNUAYCs:C',
            'BDRCVFR[feWj1Vr5u3D]': 'I67x6TjHwwYf0',
            'delPer': '0',
            'PSINO': '5',
            'BDORZ': 'B490B5EBF6F3CD402E515D22BCDA1598',
            'BDRCVFR[dG2JNJb_ajR]': 'mk3SLVN4HKm',
            'userFrom': 'null',
            'BDRCVFR[-pGxjrCMryR]': 'mk3SLVN4HKm',
            'ab_sr': '1.0.1_MTYxZDI2MWQ4MDVmNjU5Mjk5ODhkMTIxOTVjMWY2NWE4Yzk3ZGVjZjljMTE4NTM3MDM4ZDQ0NzBiNTNhODlmMGE4ODBmMzVkY2MwNTMzZmJhNDk0YzgyMWYxMzM4YjA1NWY1YTc5MzQ0NDVkNGE3MGY4OTA2ZGI1NjI0OTBiZGRlMGM3OGU5ZjY0MzBhODc4MDJmNDBjMmQzMGZjZDAwOQ==',
        }

        headers = {
            'Accept': 'text/plain, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            # 'Cookie': 'winWH=%5E6_2560x1431; BDIMGISLOGIN=0; BDqhfp=%E4%B8%9D%E5%85%89%E6%A4%8B%E9%B8%9F%26%260-10undefined%26%260%26%261; cleanHistoryStatus=0; MCITY=-315%3A; BIDUPSID=E3B4335AD1F0E6C9E9CA698B56909CBD; PSTM=1715927349; BDUSS=EF2YlRaMlozTGdTbVlzamNQVn5IZkNybVF5Mzl6N24zc3Q5ZFhkUmpwT2txWjltSVFBQUFBJCQAAAAAAAAAAAEAAADqvdA6MTM5MTVnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKQceGakHHhmS; BDUSS_BFESS=EF2YlRaMlozTGdTbVlzamNQVn5IZkNybVF5Mzl6N24zc3Q5ZFhkUmpwT2txWjltSVFBQUFBJCQAAAAAAAAAAAEAAADqvdA6MTM5MTVnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKQceGakHHhmS; BDSFRCVID=zcKOJexroG3Kj2Qtq838bQenqYHjxXjTDYrE8HDLnjRzg2_VY-fcEG0PtsBmvLtb6jBvogKKL2OTHm_F_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF=JnK8_C82tC83fP36qROE-tCshUTMetJXfb6Gsl7F5l8-hxcveJ7d3b0R2G0O0xvJyCOq_K5GbJoxOKQphTbI3tI1XaQGLhRd5GQMQxJN3KJmepC9bT3vQxuwyf8D2-biW2tH2Mbdax7rebCxD6Jab-uyM4vBbtjxB5nnV6625DnJh-PGe6L5e5oyeaKs5-7ybCPX3JjV5PK_Hn7zeP-beM4pbt-qJt7et6ndW-b9MDJSsqnmWtnJyPobXP6nBT5KaarGV4QbQx-5flIGQtbjKb_kQN3T-qLO5bRiL66Rtj6-Dn3oyUkKXp0nhJndKPujtJvD0n6b2qR8qRojXnOxeDu3bG7MJnktfJP8VCKbJID5bP365ITMMt_HbUnXKK62aJ0HKt5vWJ5TMCoLhTrWXh03Wh6UbfQ7bN5yLD0aQ4QDShPC-tP5LlLqQPkD3qbdXKTz0-jx3l02Vb3Ee-t2yUKI04rbJ4RMW23roq7mWn6r_nbzD5KbKt0DKU6raJja32OJ-IIEbnLWeIJ9jjC5Djc-DHttJTn8265KWjrtKRTffjrnhPF324tUXP6-hnjy3b7iBDolbPDWhInP0to_KqFnMl7vaq3Ry6r4VpngBnr8V4cTDln4WDuE-PoxJpOJ-2rLLqR2KDQ8E4QvbURvX5Dg3-7LyM5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIE3-oJqCKBhKPC3J; H_WISE_SIDS_BFESS=60340_60352_60361; indexPageSugList=%5B%22%E4%B8%9D%E5%85%89%E6%A4%8B%E9%B8%9F%22%2C%22%E5%86%99%E5%87%BAhelloworld%E7%9A%84%E8%A1%A8%E6%83%85%22%5D; H_WISE_SIDS=60361; BDSFRCVID_BFESS=zcKOJexroG3Kj2Qtq838bQenqYHjxXjTDYrE8HDLnjRzg2_VY-fcEG0PtsBmvLtb6jBvogKKL2OTHm_F_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF_BFESS=JnK8_C82tC83fP36qROE-tCshUTMetJXfb6Gsl7F5l8-hxcveJ7d3b0R2G0O0xvJyCOq_K5GbJoxOKQphTbI3tI1XaQGLhRd5GQMQxJN3KJmepC9bT3vQxuwyf8D2-biW2tH2Mbdax7rebCxD6Jab-uyM4vBbtjxB5nnV6625DnJh-PGe6L5e5oyeaKs5-7ybCPX3JjV5PK_Hn7zeP-beM4pbt-qJt7et6ndW-b9MDJSsqnmWtnJyPobXP6nBT5KaarGV4QbQx-5flIGQtbjKb_kQN3T-qLO5bRiL66Rtj6-Dn3oyUkKXp0nhJndKPujtJvD0n6b2qR8qRojXnOxeDu3bG7MJnktfJP8VCKbJID5bP365ITMMt_HbUnXKK62aJ0HKt5vWJ5TMCoLhTrWXh03Wh6UbfQ7bN5yLD0aQ4QDShPC-tP5LlLqQPkD3qbdXKTz0-jx3l02Vb3Ee-t2yUKI04rbJ4RMW23roq7mWn6r_nbzD5KbKt0DKU6raJja32OJ-IIEbnLWeIJ9jjC5Djc-DHttJTn8265KWjrtKRTffjrnhPF324tUXP6-hnjy3b7iBDolbPDWhInP0to_KqFnMl7vaq3Ry6r4VpngBnr8V4cTDln4WDuE-PoxJpOJ-2rLLqR2KDQ8E4QvbURvX5Dg3-7LyM5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIE3-oJqCKBhKPC3J; BAIDUID=58989DE417F26A77CAB8B758EE7097EC:FG=1; BAIDUID_BFESS=58989DE417F26A77CAB8B758EE7097EC:FG=1; H_PS_PSSID=60278_60359_60386_60426; BA_HECTOR=852h2k04a5a5252l24al01852di21e1j8crc01v; ZFY=4zNmYnZ8XUNbHKn0xqxzeaLINcR8mhO9Jv:BvlNUAYCs:C; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; delPer=0; PSINO=5; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; userFrom=null; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; ab_sr=1.0.1_MTYxZDI2MWQ4MDVmNjU5Mjk5ODhkMTIxOTVjMWY2NWE4Yzk3ZGVjZjljMTE4NTM3MDM4ZDQ0NzBiNTNhODlmMGE4ODBmMzVkY2MwNTMzZmJhNDk0YzgyMWYxMzM4YjA1NWY1YTc5MzQ0NDVkNGE3MGY4OTA2ZGI1NjI0OTBiZGRlMGM3OGU5ZjY0MzBhODc4MDJmNDBjMmQzMGZjZDAwOQ==',
            'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=result&fr=&sf=1&fmq=1720086816761_R&pv=&ic=0&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&dyTabStr=&ie=utf-8&sid=&word=%E4%B8%9D%E5%85%89%E6%A4%8B%E9%B8%9F',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'sec-ch-ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        }

        params = {
            'tn': 'resultjson_com',
            'logid': '11251570808064995215',
            'ipn': 'rj',
            'ct': '201326592',
            'is': '',
            'fp': 'result',
            'fr': '',
            'word': keyword,
            'queryWord': keyword,
            'cl': '2',
            'lm': '',
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': '-1',
            'z': '',
            'ic': '0',
            'hd': '',
            'latest': '',
            'copyright': '',
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': '0',
            'istype': '2',
            'qc': '',
            'nc': '1',
            'expermode': '',
            'nojc': '',
            'isAsync': '',
            'pn': '30',
            'rn': '30',
            'gsm': '1e',
            time.time() * 1000: '',
        }

        response = requests.get('https://image.baidu.com/search/acjson', params=params, cookies=cookies, headers=headers)
        try:
            datas = response.json()['data']
        except Exception as e:
            print(e)
            datas = json.loads(response.content.decode('utf-8').replace('\\', ''))['data']
        image_urls += [item['hoverURL'] for item in datas if 'hoverURL' in item and item['hoverURL'] != '']
        rn += 30

    # 下载图片
    count = 0
    for url in image_urls:
        # 发送请求下载图片
        img_response = requests.get(url, headers=headers)
        img_name = f"{bird_map[keyword].split('.')[1]}_{str(count+1).zfill(4)}_{random.randint(100000, 999999)}.jpg"  # 可以根据需要修改文件名
        img_path = os.path.join(download_dir, img_name)
        label = int(bird_map[keyword].split('.')[0]) - 1
        # 200.Common_Yellowthroat/Common_Yellowthroat_0040_190427.jpg 199
        content_to_append = '%s/%s %d' % (bird_map[keyword], img_name, label)
        train = np.random.binomial(n=1, p=0.5)
        if train == 1:
            file_path = train_path
        else:
            file_path = test_path
        with open(file_path, "a") as file:
            file.write(content_to_append + "\n")
        with open(img_path, 'wb') as f:
            f.write(img_response.content)
            print(f"下载图片 '{img_name}' 成功")
        count += 1


base_dir = os.path.dirname(os.path.abspath(__file__))
for k, v in bird_map.items():
    relative_dir = 'data/Bird/images/' + v
    directory = os.path.join(base_dir, relative_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    txt_dir = os.path.join(base_dir, 'data/Bird')
    search_and_download_images(k, directory, txt_dir)

directory = 'D:\\Projects\\WS_DAN_PyTorch2\\FGVC\\2022鸟图'
done = {}
for filename in os.listdir(directory):
    bird_name = filename.split('.')[0]
    if bird_name in done:
        continue
    img_dir = os.path.join(base_dir, 'data/Bird/images', bird_map[bird_name])
    all_items = os.listdir(img_dir)
    # 过滤出所有文件
    files = [item for item in all_items if os.path.isfile(os.path.join(img_dir, item))]
    count = len(files)
    img_name = f"{bird_map[bird_name].split('.')[1]}_{str(count + 1).zfill(4)}_{random.randint(100000, 999999)}.jpg"  # 可以根据需要修改文件名
    new_filename = os.path.join(img_dir, img_name)
    os.rename(os.path.join(directory, filename), new_filename)
    label = int(bird_map[bird_name].split('.')[0]) - 1
    content_to_append = '%s/%s %d' % (bird_map[bird_name], img_name, label)
    file_path = os.path.join(os.path.join(base_dir, 'data/Bird'), 'bird_test2.txt')
    with open(file_path, "a") as file:
        file.write(content_to_append + "\n")
    print(f'Renamed: {filename} -> {new_filename}')
