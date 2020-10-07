import os
import glob
import cv2
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from natsort import natsorted
from skimage import img_as_float
from skimage.metrics import structural_similarity
import networkx as nx
import requests
import shutil
import json
import sys

from clear_db_download_templates import try_download_templates_multi_thread

def get_gray_and_canny(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 50, 200)
    return img_gray, img_canny

def get_ssim_two_templates(cropped_im1, cropped_im2):
    h1,w1 = cropped_im1.shape[:2]
    h2, w2 = cropped_im2.shape[:2]

    if w1 / w2 > 1.5 or w1 / w2 < (1/1.5):
        return 0.0

    if not cropped_im2.shape == cropped_im1.shape:
        return 0.0

    ssim_res1 = structural_similarity(img_as_float(cropped_im1), img_as_float(cropped_im2), multichannel=False)
    return ssim_res1

def get_results(im1, im2):
    BLACK = [0, 0, 0]
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    # im1 = cv2.copyMakeBorder(im1, int(0.1 * h1), int(0.1 * w1), int(0.1 * h1), int(0.1 * w1), cv2.BORDER_CONSTANT,
    #                          value=BLACK)
    results = cv2.matchTemplate(im1, im2, cv2.TM_CCORR)
    _, max_val, _, max_loc = cv2.minMaxLoc(results)
    norm_max_val = max_val / (h2 * w2)
    x1,y1 = (max_loc[0], max_loc[1])
    x2,y2 = (max_loc[0] + w2, max_loc[1] + h2)

    return norm_max_val, x1, y1, x2, y2


def get_score_two_templates(im1, im2,  im1_gray, im2_gray):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    if w1 / w2 > 1.5 or w1 / w2 < (1/1.5):
        return 0.0, None, None

    if h1 >= h2 and w1 >= w2:
        norm_max_val, x1, y1, x2, y2 = get_results(im1, im2)
        cropped_im1 = im1_gray[y1:y2, x1: x2]
        cropped_im2 = im2_gray
    elif h1 <= h2 and w1 <= w2:
        norm_max_val, x1, y1, x2, y2 = get_results(im2, im1)
        cropped_im2 = im2_gray[y1:y2, x1: x2]
        cropped_im1 = im1_gray
    else:
        cropped_im1 = im1[:h2, :w2]
        cropped_im2 = im2[:h1, :w1]
        results = cv2.matchTemplate(cropped_im1, cropped_im2, cv2.TM_CCORR)
        h, w = im2.shape
        _, max_val, _, max_loc = cv2.minMaxLoc(results)
        norm_max_val = max_val / (h * w)

    return norm_max_val, cropped_im1, cropped_im2


def create_ssim_matrix(gray_list, edge_list):
    num_of_templates = len(gray_list)
    ssim_mat = np.zeros((num_of_templates, num_of_templates))

    for i in range(num_of_templates):
        print('finish template {} out of {}'.format(i, num_of_templates))
        for j in range(num_of_templates):
            if j == i:
                ssim_score = 0
            elif i > j:
                continue
            else:
                norm_max_val, cropped_im1, cropped_im2 = get_score_two_templates(edge_list[i], edge_list[j], gray_list[i], gray_list[j])
                if norm_max_val <= 0:
                    ssim_score = 0
                else:
                    ssim = get_ssim_two_templates(cropped_im1, cropped_im2)
                    ssim_score = norm_max_val * ssim
            ssim_mat[j, i] = ssim_score
            ssim_mat[i, j] = ssim_score

    return ssim_mat

def find_best_matches(ssim_mat, th=1200):
    max_val_list = []
    max_loc_list = []
    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(ssim_mat)
        if max_val < th:
            break
        max_loc = list(max_loc)
        x,y = max_loc
        max_val_list.append(max_val)
        max_loc_list.append(max_loc)
        ssim_mat[x,y] = 0
        ssim_mat[y,x] = 0

    return max_loc_list, max_val_list


def split_to_connected_connected_components(max_loc_list, templates_list,templates_folder):
    flat_list = np.unique([item for sublist in max_loc_list for item in sublist])
    G = nx.Graph()
    G.add_edges_from(max_loc_list)
    # nx.draw(G)
    # plt.show()
    edges_dict = {}
    for i in flat_list:
        edges_dict[os.path.basename(templates_list[i])] = len(G.edges(i))
    connected_components = nx.connected_components(G)
    connected_components = sorted(connected_components, key=lambda x: len(x), reverse=True)
    for j, connected_component in enumerate(connected_components):
        copy_folder = os.path.join(templates_folder, str(j))
        os.makedirs(copy_folder, exist_ok=True)
        for val in connected_component:
            shutil.copy2(templates_list[val], copy_folder)
    for val in range(len(templates_list)):
        if val not in flat_list:
            copy_folder = os.path.join(templates_folder, 'other')
            os.makedirs(copy_folder, exist_ok=True)
            shutil.copy2(templates_list[val], copy_folder)
    return edges_dict


def get_best_template_in_folder(folder, edges_dict):
    templates_list = glob.glob(folder + '/*.png')
    templates_list = natsorted(templates_list, key=lambda y: y.lower())
    num_of_templates = len(templates_list)
    num_of_best_templates = int(np.ceil(num_of_templates / 30))
    sum_list = []
    edges_list = []
    for template in templates_list:
        base_name = os.path.basename(template)
        img = cv2.imread(template)
        try:
            _ , img_canny = get_gray_and_canny(img)
        except:
            continue
        img_canny = np.asarray(img_canny)
        sum_edges = img_canny.sum()
        sum_list.append(sum_edges)
        edges_list.append([template, sum_edges, edges_dict[base_name]])
    max_edges = np.asarray(sorted([i[2] for i in edges_list], reverse=True))
    indexes = np.unique(max_edges, return_index=True)[1]
    max_edges_unique = [max_edges[index] for index in sorted(indexes)]
    sorted_list = []
    for max_ in max_edges_unique:
        max_edges_list = [i for i in edges_list if i[2] == max_]
        sorted_list.extend(sorted(max_edges_list, key=lambda x: x[1], reverse=True))

    return [sorted_list[i][0] for i in range(num_of_best_templates)]


def remove_png(template_name):
    return template_name.replace(".png", "")


def get_templates_in_folder(folder):
    template_list = os.listdir(folder)
    new_list = []
    for template in template_list:
        if not template.endswith('.png'):
            continue
        new_list.append(template)
    return new_list


def decide_best_template(main_folder, edges_dict, target_folder):
    new_folder = os.path.join(main_folder, 'new')
    os.makedirs(new_folder, exist_ok=True)
    if target_folder is not None:
        os.makedirs(target_folder, exist_ok=True)
    folder_list = os.listdir(main_folder)
    filter_db_dict = {}
    for _folder in folder_list:
        folder = os.path.join(main_folder, _folder)
        if not os.path.isdir(folder):
            continue
        if _folder == 'bad':
            templates_in_folder = get_templates_in_folder(folder)
            for template in templates_in_folder:
                filter_db_dict[remove_png(template)] = 'bad_template'
        elif _folder == 'other':
            templates_in_folder = get_templates_in_folder(folder)
            for template in templates_in_folder:
                shutil.copy2(os.path.join(folder, template), new_folder)
                filter_db_dict[remove_png(template)] = remove_png(template)
        elif _folder == 'new':
            continue
        else:
            best_templates = get_best_template_in_folder(folder, edges_dict)
            for template in best_templates:
                shutil.copy2(template, new_folder)
            templates_in_folder = get_templates_in_folder(folder)
            for template in templates_in_folder:
                filter_db_dict[remove_png(template)] = remove_png(os.path.basename(best_templates[0]))
        # with open(os.path.join(main_folder, 'old2newTemplates.json'), 'w') as outfile:
        #     json.dump(filter_db_dict, outfile)

    all_templates = get_templates_in_folder(main_folder)
    for template in all_templates:
            filter_db_dict[remove_png(template)]
    return filter_db_dict
        # for b in best_templates:
        #     shutil.copy2(b, new_folder)
        #     if target_folder is not None:
        #         shutil.copy2(b, target_folder)


def filter_folder(templates_folder, target_folder=None):
    bad_folder = os.path.join(templates_folder, 'bad')
    templates_list = get_templates_in_folder(templates_folder)

    templates_list = natsorted(templates_list, key=lambda y: y.lower())
    new_template_list = []
    gray_list, edge_list = [], []
    for img_path in templates_list:
        img_path = os.path.join(templates_folder, img_path)
        img = cv2.imread(img_path)
        try:
            img_gray, im_canny = get_gray_and_canny(img)
            h, w = img.shape[:2]
        except:
            h, w = 0, 0
        if h * w < 3000 or np.sum(im_canny) / (h * w) < 10.0:
            os.makedirs(bad_folder, exist_ok=True)
            shutil.copy2(img_path, bad_folder)
        else:
            new_template_list.append(img_path)
            gray_list.append(img_gray)
            edge_list.append(im_canny)
    ssim_mat = create_ssim_matrix(gray_list, edge_list)
    # plt.imshow(ssim_mat)
    # plt.show()
    max_loc_list, max_val_list = find_best_matches(ssim_mat, th=1000)
    edges_dict = split_to_connected_connected_components(max_loc_list, new_template_list, templates_folder)
    filter_db_dict = decide_best_template(templates_folder, edges_dict, target_folder)
    return filter_db_dict

def clear_folder(main_folder):
    files_list = glob.glob(main_folder + '/*.json')
    for file in files_list:
        os.remove(file)
    folder_list = os.listdir(main_folder)
    if len(folder_list) == 0:
        shutil.rmtree(main_folder, ignore_errors=True)
        return
    for i, _folder in enumerate(folder_list):
        folder = os.path.join(main_folder, _folder)
        if not os.path.isdir(folder):
            continue
        shutil.rmtree(folder)

def copy_filterd_round(main_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    folder_list = os.listdir(main_folder)
    if 'new' in folder_list:
        folder = os.path.join(main_folder, 'new')
        files = os.listdir(folder)
        for file in files:
            if not file.endswith(".png"):
                continue
            shutil.copy2(os.path.join(folder, file), target_folder)
        return False
    else:
        return True

def main():
    target_folder = None
    if sys.platform == "linux" or sys.platform == "linux2":
        main_folder = r'/data/home/wscuser/algorithms/boxscore_to_text/templates_search_all'
    else:
        main_folder = r'C:\Data\BSL\templates_search'
    all_rounds_db = {}
    folder_list = os.listdir(main_folder)
    folder_list = natsorted(folder_list, key=lambda y: y.lower())
    for i, _folder in enumerate(folder_list):
        folder = os.path.join(main_folder, _folder)
        if not os.path.isdir(folder):
            continue
        # flag = copy_filterd_round(folder, target_folder)
        clear_folder(folder)
        filter_db_dict = filter_folder(folder, target_folder)
        all_rounds_db[_folder] = filter_db_dict
        with open(os.path.join(main_folder, 'old2newTemplates.json'), 'w') as outfile:
            json.dump(all_rounds_db, outfile)

        print('finish folder {} {} out of {}'.format(_folder, i, len(folder_list)))

def count_new_db():
    main_folder = r'.\templates_search'
    folder_list = os.listdir(main_folder)
    folder_list = natsorted(folder_list, key=lambda y: y.lower())
    templates_list_orig = []
    templates_list_new = []
    for i, _folder in enumerate(folder_list):
        folder = os.path.join(main_folder, _folder)
        if not os.path.isdir(folder):
            continue
        templates_list_orig.extend(glob.glob(folder + '\*.png'))
        templates_list_new.extend(glob.glob(folder + r'\new\*.png'))
    print(len(templates_list_new)/ len(templates_list_orig))

def copy_all_templates():
    target_folder = r'C:\Data\BSL\templates_search_all\0'
    main_folder = r'C:\Data\BSL\templates_search'
    folder_list = os.listdir(main_folder)
    folder_list = natsorted(folder_list, key=lambda y: y.lower())
    templates_list_orig = []
    templates_list_new = []
    for i, _folder in enumerate(folder_list):
        folder = os.path.join(main_folder, _folder)
        if not os.path.isdir(folder):
            continue
        templates_list_orig.extend(glob.glob(folder + '\*.png'))
    for file in templates_list_orig:
        target_path = os.path.join(target_folder, os.path.basename(file))
        if os.path.isfile(target_path):
            target_size = os.path.getsize(target_path)
            source_size = os.path.getsize(file)
            if not target_size == source_size and target_size > 0 and source_size > 0:
                print(target_path)
                print(file)
            if target_size == 0:
                shutil.copy2(file, target_folder)
        else:
            shutil.copy2(file, target_folder)
    print(len(os.listdir(target_folder)))


def create_database():
    json_ = r"C:\Data\BSL\templates_search\goodTemplatesDict.json"
    json_out = r"C:\Data\BSL\templates_search\goodTemplatesDictID.json"
    with open(json_) as json_file:
        data = json.load(json_file)
    counter = 0
    data_out = {}
    for roundID, templates_list in data.items():
        image_dict = {}
        try:
            print("start db {} out of {}".format(counter, len(data)))
            system_path = os.path.join(box_score_db, roundID) + '?returnAllElements=true'
            response = requests.get(system_path)
            box_score_elements = response.json()
            for key, value in box_score_elements.items():
                image_id = value['ImageId']
                elements = value['BoxscoreElements']
                image_name = image_id + '.png'
                image_dict[image_id] = key
            temp_dict = {}
            for key, value in templates_list.items():
                temp_list = [image_dict[remove_png(template)] for template in value]
                temp_dict[key] = temp_list

                # if value == 'bad_template':
                #     temp_dict[image_dict[key]] = 'bad_template'
                # else:
                #     temp_dict[image_dict[key]] = image_dict[value]
            data_out[roundID] = temp_dict
        except:
            pass
        counter += 1
    with open(json_out, 'w') as outfile:
        json.dump(data_out, outfile)


def filtered_db_to_good_templates(templates_round_folder, image_dict):
    round_id = os.path.basename(templates_round_folder)
    good_templates_dict = {}

    good_templates_list = get_templates_in_folder(os.path.join(templates_round_folder, 'new'))
    bad_templates_list = [template for template in get_templates_in_folder(templates_round_folder) if template not in good_templates_list]

    good_templates_list = [image_dict[template.replace(".png", "")]['boxscoreID'] for template in good_templates_list]
    bad_templates_list = [image_dict[template.replace(".png", "")]['boxscoreID'] for template in bad_templates_list]

    good_templates_dict[round_id] = {'goodTemplates':good_templates_list, 'badTemplates':bad_templates_list}

    json_dict = os.path.join(templates_round_folder, 'goodTemplatesDict_{}.json'.format(round_id))
    with open(json_dict, 'w') as outfile:
        json.dump(good_templates_dict, outfile)
    print(good_templates_dict)
    return json_dict

def filter_round(round_id, env):
    if env == 'UAT':
        box_score_db = r'https://wscalgosite-uat-op.azurewebsites.net/Boxscores'
    elif env == 'PROD':
        box_score_db = r'https://wscalgosite-op.azurewebsites.net/Boxscores'
    templates_folder = './templates_folder'
    os.makedirs(templates_folder, exist_ok=True)
    images_url = 'https://wsczoominnba.blob.core.windows.net/boxscores/'
    templates_round_folder, image_dict = try_download_templates_multi_thread(round_id, templates_folder, box_score_db, images_url)
    filter_folder(templates_round_folder)
    filtered_db_to_good_templates(templates_round_folder, image_dict)
    shutil.rmtree(templates_folder)


if __name__ == "__main__":
    # main()
    # create_database()
    parser = ArgumentParser()
    parser.add_argument('-env', help='choose UAT or PROD')
    parser.add_argument('-round_id', type=int)
    args = parser.parse_args()
    env = args.env
    round_id = str(args.round_id)
    filter_round(round_id, env)
