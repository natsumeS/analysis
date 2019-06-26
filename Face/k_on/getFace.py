import Face.k_on as kon


def get_face_data():
    # 画像の保存フォルダを指定
    raw_dir = "/recognition/k-on/raw"
    gagent = kon.GoogleSearchAgent(raw_dir)

    # 検索ワードとカテゴリを追加
    keyword = ["平沢唯", "田井中律", "秋山澪", "琴吹紬", "中野梓"]
    for i in range(5):
        gagent.add_search(keyword[i], kon.cat[i])

    # img_numでダウンロードするファイル数を指定（最大80)
    gagent.search(img_num=80)

    # 　画像検索で取得した画像から顔を切り取ります。
    kon.faceagent.cut_face(raw_dir)


if __name__ == "__main__":
    get_face_data()
