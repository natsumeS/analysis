import Face.k_on as kon


def learning():
    kon.faceagent.learning(200, kon.optimizer, 500, 100, plot_show=True)


if __name__ == "__main__":
    learning()
