import base64
import re
import tempfile
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from os import path
from pathlib import Path
import requests

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7,es;q=0.6,he;q=0.5,la;q=0.4,pl;q=0.3",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "servicos.receita.fazenda.gov.br",
    "Pragma": "no-cache",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3928.4 Safari/537.36"
}

REGEX_IMGDATA = re.compile("imgCaptcha.*?src=\"(.*?)\"", flags=re.MULTILINE)


def download_image():
    try:
        conteudo = requests.get(
            "https://servicos.receita.fazenda.gov.br/Servicos/CPF/ConsultaSituacao/ConsultaPublicaSonoro.asp",
            headers=HEADERS, verify=False).text
    except:
        return False

    data_image = REGEX_IMGDATA.findall(conteudo)[0]

    if 'data:image/png;base64,' in data_image:
        base64_data = data_image[22:]
        data = base64.b64decode(base64_data)
        with open( Path("./data/originais") / (str(uuid.uuid4()) + '.png'), "wb") as arquivo:
            arquivo.write(data)
        return True
    return False


def download_mult_images(quantidade = 10):

    while quantidade > 0:
        baixou = download_image()
        if baixou:
            quantidade -= 1


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(100):
            executor.submit(download_mult_images, 100)