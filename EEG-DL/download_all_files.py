# based on https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/MIND_Get_EDF.py # noqa
import requests
from bs4 import BeautifulSoup
from multiprocessing.pool import ThreadPool


DATA_DIRECTORY = '_data'
NUM_SUBJECTS = 109

# Be careful with this value. Your connections will get
#   cut if you have too many for too long, plus it's
#   rude to tie up the sites bandwidth too much.
CONCURRENT_REQUESTS = 3


# Download file to working directory based on given url
def download_file(url):
    print("Downloading " + url)
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(DATA_DIRECTORY+"/"+local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
    return local_filename


# Parse a elements in html, add them to links array if they end in .edf
def extract_links_from_html(html, links, root_link):
    soup = BeautifulSoup(html.text, features="html.parser")

    for link in soup.find_all('a'):
        link_href = link.get('href')
        if link_href.endswith(".edf"):
            links.append(root_link+link_href)


# Return string array of links to download
def get_links():
    links = []
    for i in range(1, NUM_SUBJECTS+1):
        print("Getting links for subject {0}".format(i))
        root_link = ("https://archive.physionet.org/pn4/eegmmidb/S{0:03}/"
                     .format(i))
        response = requests.get(root_link)

        if (response.status_code == 200):
            extract_links_from_html(response, links, root_link)
        else:
            print("Failed to retrieve links for subject {0}"
                  .format(i))

    return links


links = get_links()
results = ThreadPool(CONCURRENT_REQUESTS).imap_unordered(download_file, links)

for result in results:
    print("Finished downloading " + result)
'''
for link in links:
    download_file(link)
'''

print("Downloads complete.")
