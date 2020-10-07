import os
from threading import Thread
from queue import Queue
from queue import Empty as Queue_Empty
import logging
from urllib.error import HTTPError, URLError
import urllib.request
import requests
import shutil
failed_downloads = []


dl_queue = Queue()
def download_file(url, dir=''):
    """
    download a file in url to dir with the original file name in the url
    :param url: url to download
    :type url: str
    :param dir: dir to download the file to, if left empty then the file will be placed at pwd
    :type dir: str
    :return: download success, path to file
    """
    file_name = os.path.join(dir, url.split('/')[-1])

    logging.info('starting download of {}\n'.format(url))

    if os.path.exists(file_name):
        logging.info(file_name + ' already exists; not downloading')
        return True, file_name

    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        data = response.read()  # a `bytes` object
        out_file.write(data)

    return True, file_name

def download_files(dst=''):

    if dst:
        os.makedirs(dst, exist_ok=True)

    while not dl_queue.empty():
        try:
            urls = dl_queue.get(block=False)
            file_names = []
            for url in urls:
                try:
                    # dst = os.path.join(dst, os.path.basename(url))
                    success, file_name = download_file(url, dst)
                except (HTTPError, URLError) as dl_error:
                    success, file_name = False, ''
                if success:
                    file_names.append(file_name)
                else:
                    failed_downloads.append(file_name)
            dl_queue.task_done()
        except Queue_Empty:
            logging.info("Queue empty")


def download_templates_multi_thread(save_folder, urls_list, number_of_threads=15):
    os.makedirs(save_folder, exist_ok=True)
    if len(urls_list) == 0:
        return
    for url in urls_list:
        dl_queue.put([url])
    all_threads = []
    for i in range(min(len(urls_list), number_of_threads)):
        dl_thread = Thread(target=download_files, args=([save_folder]))
        all_threads.append(dl_thread)
        dl_thread.start()

    for dl_thread in all_threads:
        dl_thread.join()


def try_download_templates_multi_thread(round_id, templates_folder, box_score_db, images_url):

    templates_round_folder = os.path.join(templates_folder, round_id)
    if os.path.isdir(templates_round_folder):
        shutil.rmtree(templates_round_folder, ignore_errors=True)
    os.makedirs(templates_round_folder, exist_ok=True)
    system_path = os.path.join(box_score_db, round_id) + '?returnAllElements=true'
    response = requests.get(system_path)
    box_score_elements = response.json()
    if box_score_db is None:
        return None
    box_score_elements = box_score_elements['boxscoresDictionary']
    existing_files = len(os.listdir(templates_round_folder))
    image_dict = {}
    urls_list = []
    for key, value in box_score_elements.items():
        image_id = value['imageId']
        image_name = image_id + ".png"
        image_src = os.path.join(images_url, image_name)
        image_dest = os.path.join(templates_round_folder, image_name)
        elements = value['boxscoreElements']
        # elements = move_elements_for_debug(elements)
        image_dict[image_id] = {'boxscoreID':key, 'elements':elements}

        if os.path.isfile(image_dest):
            continue
        urls_list.append(image_src)
    print('start downloading templates')
    download_templates_multi_thread(templates_round_folder, urls_list)
    print('finish downloading templates')
    templates_files_list = os.listdir(templates_round_folder)
    counter = len(templates_files_list) - existing_files
    # print('counter is {}'.format(counter))
    for temp in templates_files_list:
        temp_file = os.path.join(templates_round_folder, temp)
        if temp.replace(".png", "") not in image_dict.keys() and os.path.isfile(temp_file):
            os.remove(temp_file)

    return templates_round_folder, image_dict


