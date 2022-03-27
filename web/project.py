import os


class Project:
    def __init__(self,file,path):
        self.file = file
        self.file_path = path
        self.create_folder()


    def get_file_path(self):
        return self.file_path

    def create_folder(self):

        self.file_path = self.uniquify(self.file_path)
         
        self.ensure_dir(self.file_path)

    def vid_to_frames(self):
        print("bello")

    @staticmethod
    def uniquify(path):
        filename, extension = os.path.splitext(path)
        counter = 1
        while os.path.exists(path):
            path = filename + " (" + str(counter) + ")" + extension
            counter += 1
        return path

    @staticmethod
    def ensure_dir(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)