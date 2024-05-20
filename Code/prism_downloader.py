import io
import time
import datetime
import random
import zipfile

import requests

start_dt = datetime.date(2009, 3, 7)
end_dt = datetime.date(2022, 12, 31)
delta = datetime.timedelta(days=1)

dtype = 'tmean'
domain = f'https://ftp.prism.oregonstate.edu/daily/{dtype}'

while start_dt <= end_dt:
    dt = start_dt.strftime('%Y%m%d')
    f = f'PRISM_{dtype}_stable_4kmD2_{dt}_bil.zip'
    url = f'{domain}/{start_dt.year}/{f}'
    print(url)
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(r'./DATA')
    start_dt += delta
    time.sleep(random.randint(1, 4))

