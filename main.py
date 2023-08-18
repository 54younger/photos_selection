import os
from PIL import Image
from progressbar import ProgressBar
import time

import shutil
pbar = ProgressBar()


class Photos:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.screenshot = []
        self.anime = []
        self.photo = []
        self.others = []
        self.get_photos()
    def if_message_exist(self,filename):
        with Image.open(os.path.join(self.folder_path,filename)) as image:
            if not (image._getexif()):
                return False
            elif (image._getexif().get(0x829A)):
                return True
            else:
                return False
    def get_photos(self):
        print("正在获取并预处理照片...\n")
        if not os.path.exists(os.path.join(self.folder_path,"photo")):
            os.mkdir(os.path.join(self.folder_path,"photo"))
        if not os.path.exists(os.path.join(self.folder_path,"screenshot")):
            os.mkdir(os.path.join(self.folder_path,"screenshot"))
        if not os.path.exists(os.path.join(self.folder_path,"anime")):
            os.mkdir(os.path.join(self.folder_path,"anime"))
        for filename in pbar(os.listdir(self.folder_path)):
            if (filename.endswith(".JPEG") or filename.endswith(".JPG")):
                if(self.if_message_exist(filename)):
                    #is photo
                    self.photo.append(filename)
                    shutil.move(os.path.join(self.folder_path,filename),os.path.join(self.folder_path,"photo",filename))
                else:
                    #is anime
                    self.anime.append(filename)
                    shutil.move(os.path.join(self.folder_path,filename),os.path.join(self.folder_path,"anime",filename))
            elif (filename.endswith(".png") or filename.endswith(".PNG")):
                #is screenshot
                self.screenshot.append(filename)
                shutil.move(os.path.join(self.folder_path,filename),os.path.join(self.folder_path,"screenshot",filename))
            elif (filename.endswith(".MOV") or filename.endswith(".MP4")) :
                os.remove(os.path.join(self.folder_path, filename))
            else:
                #is others
                self.others.append(filename)
            time.sleep(0.01)
        print("照片处理完成！\n")

Photos("C:\\Users\\younger\\Pictures\\ApplePic")
