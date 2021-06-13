
from lib.datasets.pascal_voc import pascal_voc
from lib.utils.config import Config as config
import matplotlib.pyplot as plt


def vis(img, bboxes, labels):
    """
    这个函数用来看看一个样本的图形、bbox、对应的类别，和scale大小。
    """
    # img = img.numpy() #[-1,1]
    # img = (img * 0.225) + 0.45
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

    for i in range(len(bboxes)):
        x1 = bboxes[i][0]
        y1 = bboxes[i][1]
        x2 = bboxes[i][2]
        y2 = bboxes[i][3]
        height = y2 - y1
        width = x2 - x1
        ax.add_patch(
            plt.Rectangle((x1, y1),
                          width,
                          height,
                          fill=False,
                          edgecolor='red',
                          linewidth=2))
        ax.text(x1,
                y1,
                config.classes[labels[i]],
                style='italic',
                bbox={
                    'facecolor': 'white',
                    'alpha': 0.5,
                    'pad': 0
                })
    plt.show()
    return ax


if __name__ == "__main__":
    config = config()
    config.display()
    data = pascal_voc(config.image_set)
    print(len(data))
    # image_path = data.image_path_at(0)
    # # print(image_path)
    # image = tf.io.read_file(image_path)
    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.cast(image, dtype=tf.float32)
    # image = image / 255.0
    # # image = (image - tf.constant([0.485, 0.456, 0.406])) / tf.constant([0.229, 0.224, 0.225])
    # bboxes, labels = data._load_pascal_annotation(0)
    # print(bboxes, labels)

    # vis(image, bboxes, labels)