import Face.k_on as kon


def use():
    # 学習済みニューラルネットを読み込む
    kon.faceagent.load_model("/recognition/k-on/output/best_classifier.npz")

    # 正解率
    print(kon.faceagent.get_accuracy_rate())

    # AVIファイルで顔識別
    kon.faceagent.put_face_avi("/recognition/k-on/video/op.avi", "/recognition/k-on/video/op-face.avi", fps_rating=0.5,
                               size_rating=0.5)


if __name__ == "__main__":
    use()
