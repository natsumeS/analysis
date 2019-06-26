from DeepLearning import *
from Face.googlesearch import GoogleSearchAgent
from Face.face_recognition import  FaceRecognitionAgent

'''
input:image_size=64*64*3
conv1:64*64*3->60*60*10
pooling:60*60*10->30*30*10
conv2:30*30*10->26*26*5
pooling:26*26*5->13*13*5
full_layer:13*13*5->200
output:200->5
'''
cat = ["yui", "ritsu", "mio", "mugi", "azusa"]
cat_color = [(39, 27, 217), (68, 165, 253), (214, 169, 36), (170, 88, 231), (151, 176, 54)]
cascade_url = "lbpcascade_animeface.xml"


class kon_net(Chain):
    def __init__(self):
        super().__init__(
            c1=L.Convolution2D(3, 10, 5),
            c2=L.Convolution2D(10, 5, 5),
            l1=L.Linear(845, 200),
            l2=L.Linear(200, 5))

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.c1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.c2(h)), 2)
        h = F.relu(self.l1(h))
        return self.l2(h)

model = kon_net()
optimizer = optimizers.Adam(alpha=1.0e-3)
faceagent = FaceRecognitionAgent(cat, "/recognition/k-on/", cascade_url, model,
                                 color_list=cat_color)
