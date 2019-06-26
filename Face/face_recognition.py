from DeepLearning import *
import os
import sys
import glob
import random
from PIL import Image
import cv2
import shutil


# 画像の回転
def rotate(image, r):
    h, w, ch = image.shape  # 画像の配列サイズ
    M = cv2.getRotationMatrix2D((w / 2, h / 2), r, 1)  # 画像を中心に回転させるための回転行列
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


'''
顔認識クラス(FaceRecognitionAgent)
__init__(category_list,dir_n,cascade_classifier_file,model,*,color_list=None
category_list:分類クラス名のリスト
dir_n:ファイル生成のディレクトリ名
cascade_classifier_file:カスケードファイルのパス
model:chainerによって生成されたモデル
color_list:分類クラスの色のリスト（任意）
'''


class FaceRecognitionAgent:
    # initialize
    def __init__(self, category_list, dir_n, cascade_classifier_file, model, *, color_list=None):
        self.category_list = category_list
        self.color_list = color_list
        self.dir_n = dir_n
        if not os.path.exists(self.dir_n):
            os.makedirs(self.dir_n)
        self.model = model
        self.classifier = cv2.CascadeClassifier(cascade_classifier_file)

    '''
    cut_face(cat_image_dir_list)
    画像データから顔を切り出してcat/分類クラス名/に格納します。
    ノイズ等は自分で取り除くこと
    cat_image_dir_list:画像データのあるフォルダのディレクトリあるいはリスト（分類クラスのリストの順で）
    前者の場合、各クラスのフォルダの親フォルダ、後者の場合、各クラスのフォルダ
    のパスを指定
    '''

    def cut_face(self, cat_image_dir):
        for cat_id in range(len(self.category_list)):
            if not os.path.exists("{}/cat/{}".format(self.dir_n, self.category_list[cat_id])):
                os.makedirs("{}/cat/{}".format(self.dir_n, self.category_list[cat_id]))
            if isinstance(cat_image_dir, list):
                ci_url = cat_image_dir[cat_id]
            else:
                ci_url = "{}/{}".format(cat_image_dir, self.category_list[cat_id])
            files = glob.glob(os.path.join(ci_url, "*.jpg"))
            counter = 0
            for file in files:
                img = cv2.imread(file)
                for r in range(-12, 13, 4):
                    img_r = rotate(img, r)
                    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
                    faces = self.classifier.detectMultiScale(gray)
                    for (x, y, w, h) in faces:
                        dst = img_r[y:y + h, x:x + w]
                        dst = dst[::, :, ::-1]
                        resize = Image.fromarray(dst)
                        resize = resize.resize((64, 64), Image.ANTIALIAS)
                        resize.save("{}/cat/{}/{}__{}.jpg".format(self.dir_n, self.category_list[cat_id],
                                                                  self.category_list[cat_id], counter), "JPEG",
                                    quality=100,
                                    optimize=True)
                        counter += 1
            print("{}'s face is cut".format(self.category_list[cat_id]))

    '''
    divide_data(teach_cat_data_num)
    データを学習データとテストデータに分ける。
    teach_Cat_data_num:それぞれのクラスから学習データをいくつ取り出すか
    '''

    def divide_data(self, teach_cat_data_num):
        cat_data_dir_list = []
        for cat_name in self.category_list:
            cat_data_dir_list.append("{}/cat/{}".format(self.dir_n, cat_name))
        if os.path.exists("{}/teach".format(self.dir_n)):
            shutil.rmtree("{}/teach".format(self.dir_n))
        os.makedirs("{}/teach".format(self.dir_n))
        if os.path.exists("{}/test".format(self.dir_n)):
            shutil.rmtree("{}/test".format(self.dir_n))
        os.makedirs("{}/test".format(self.dir_n))
        for cat_data_dir in cat_data_dir_list:
            files = glob.glob(os.path.join(cat_data_dir, "*.jpg"))
            assert len(files) > teach_cat_data_num, "lack of category data"
            random.shuffle(files)
            teach_dataset = files[:teach_cat_data_num]
            test_dataset = files[teach_cat_data_num:]
            for file in teach_dataset:
                img = Image.open(file)
                img.save("{}/teach/{}".format(self.dir_n, os.path.basename(file)))
            for file in test_dataset:
                img = Image.open(file)
                img.save("{}/test/{}".format(self.dir_n, os.path.basename(file)))

    '''
    get_catid_from_datafile(file)
    ファイル名からクラスidを返す。クラスidはクラス配列のindexに対応
    file:ファイル名
    ファイル名は" クラス名__***"にしてください。***はワイルドカード
    '''

    def get_catid_from_datafile(self, file):
        bfn = os.path.basename(file)
        name, _ = bfn.split("__")
        for cat_id in range(len(self.category_list)):
            if name == self.category_list[cat_id]:
                return cat_id
        assert False, "name of datafile is not valid. img filename is correspond to category name"

    '''
    learning_step(optimizer, epoch_max, dataset_num, *, plot_show=True)
    学習を行います。
    optimizer:降下法（chainerのoptimizer)
    epoch_max:最大エピソード
    datast_num:1エピソードで用いる学習データの数
    plot_show=True:別ウィンドウに学習状況を表示するか
    '''

    def learning_step(self, optimizer, epoch_max, dataset_num, *, plot_show=True):
        optimizer.setup(self.model)
        pla = plotagent("{}/output".format(self.dir_n), "loss", "loss", plot_show=plot_show)
        epoch = 0
        best_loss = 10000
        teach_dir = "{}/teach".format(self.dir_n)
        pla.csv_recode_start(10000)
        while epoch < epoch_max:
            files = random.sample(glob.glob(os.path.join(teach_dir, "*.jpg")), dataset_num)
            teach_data = []
            output_data = []
            for file in files:
                img = Image.open(file)
                teach_data.append(np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255)
                output_data.append(self.get_catid_from_datafile(file))
            loss = F.softmax_cross_entropy(self.model(np.array(teach_data, dtype=np.float32)),
                                           np.array(output_data, dtype=np.int32))
            self.model.cleargrads()
            loss.backward()
            optimizer.update()
            epoch += 1
            pla.add_data(epoch, loss.data)
            pla.csv_add_data(epoch, loss.data)
            if loss.data < best_loss:
                pla.save_model("best_classifier", self.model)
                best_loss = loss.data

    '''
    正解率の計算
    '''

    def get_accuracy_rate(self):
        test_dir = "{}/test".format(self.dir_n)
        correct = 0
        test_data_num = 0
        files = glob.glob(os.path.join(test_dir, "*.jpg"))
        for file in files:
            img = Image.open(file)
            output = \
                F.softmax(self.model(np.array([np.asarray(img).transpose(2, 0, 1).astype(np.float32) / 255]))).data[0]
            cat_id = np.argmax(output)
            if cat_id == self.get_catid_from_datafile(file):
                correct += 1
            test_data_num += 1
        return correct / test_data_num

    '''
    モデルの呼び出し
    '''

    def load_model(self, npz_file=None):
        if npz_file is None:
            npz_file = "{}/output/best_classifier.npz".format(self.dir_n)
        serializers.load_npz(npz_file, self.model)

    '''
    学習
    '''

    def learning(self, teach_data_cut_num, optimizer, epoch_max, dataset_num, *, plot_show=True):
        self.divide_data(teach_data_cut_num)
        print("divide data")
        self.learning_step(optimizer, epoch_max, dataset_num, plot_show=plot_show)
        self.load_model()
        print(self.get_accuracy_rate())
        shutil.rmtree("{}/teach".format(self.dir_n))
        shutil.rmtree("{}/test".format(self.dir_n))

    '''
    put_face(img)
    画像から顔認識を行い、顔に枠を表示
    img:画像あるいは画像のリスト
    注意：リストの場合、一番最初の画像で顔認識を行い、顔の枠を表示。
    残りの画像には、その枠をprintする。
    '''

    def put_face(self, img):
        if not isinstance(img, list):
            img = [img]
        gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            dst = img[0][y:y + h, x:x + w]
            dst = dst[::, :, ::-1]
            face_input = Image.fromarray(dst)
            face_input = face_input.resize((64, 64), Image.ANTIALIAS)
            output = \
                F.softmax(
                    self.model(
                        np.array([np.asarray(face_input).transpose(2, 0, 1).astype(np.float32) / 255]))).data[0]
            cat_id = np.argmax(output)
            for img_ele in img:
                if self.color_list is None:
                    cv2.rectangle(img_ele, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                    cv2.rectangle(img_ele, (x, y), (x + w, y + h), self.color_list[cat_id], 2)
                cv2.rectangle(img_ele, (x, y - 15), (x + w, y), (0, 0, 0), -1)
                cv2.putText(img_ele, "{}:{}".format(self.category_list[cat_id], output[cat_id]), (x, y - 2),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1, cv2.LINE_AA)

    '''
    put_face_avi
    aviファイルに顔認識を適応し、枠を表示します。
    学習済みmodelを読み込んでおいてください。
    avi_file:対称aviファイルのパス
    save_avi:保存するaviファイルのパス
    fpx_rating:対象aviのfps*fps_rating
    size_rating:対象aviファイルのサイズ*size_rating
    '''

    def put_face_avi(self, avi_file, save_avi, *, fps_rating=1.0, size_rating=1.0):
        org = cv2.VideoCapture(avi_file)
        FPS = org.get(cv2.CAP_PROP_FPS) * fps_rating
        FOURCC = cv2.VideoWriter_fourcc(*'ULRG')
        W = int(org.get(cv2.CAP_PROP_FRAME_WIDTH) * size_rating)
        H = int(org.get(cv2.CAP_PROP_FRAME_HEIGHT) * size_rating)
        out = cv2.VideoWriter(save_avi, FOURCC, FPS, (W, H))
        img_list = []
        end_flag, c_frame = org.read()
        counter = 0
        while end_flag == True:
            c_frame = cv2.resize(c_frame, (W, H))
            img_list.append(c_frame)
            if len(img_list) > 2:
                self.put_face(img_list)
                for img in img_list:
                    out.write(img)
                img_list = []
                sys.stdout.write("\r{}/{}".format(counter, org.get(cv2.CAP_PROP_FRAME_COUNT)))
                sys.stdout.flush()
            end_flag, c_frame = org.read()
            counter += 1
        org.release()
        out.release()
