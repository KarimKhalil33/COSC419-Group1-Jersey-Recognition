import os
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

dl = SNdl(LocalDirectory='data/SoccerNet')

dl.downloadDataTask(task='jersey-2023', split=['train'])

