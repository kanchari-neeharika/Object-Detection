# ============================================================
# 1. IMPORTING LIBRARIES
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================
IM_WIDTH = 75
IM_HEIGHT = 75
USE_NORMALIZED_COORDINATES = True
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_CSV = "mnist_train.csv"
TEST_CSV = "mnist_test.csv"

# ============================================================
# 2. VISUALIZATION UTILITIES
# ============================================================
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,
                                color="red", thickness=1,
                                display_str_list=()):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if USE_NORMALIZED_COORDINATES:
        left, right, top, bottom = (xmin * im_width, xmax * im_width,
                                     ymin * im_height, ymax * im_height)
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax

    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)


def draw_bounding_boxes_on_image(image, boxes, color=(), thickness=1,
                                  display_str_list=()):
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError("Input must be of size [N, 4]")
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(
            image, boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
            color[i] if color else "red", thickness,
            display_str_list[i] if display_str_list else ()
        )


def draw_bounding_boxes_on_image_array(image, boxes, color=(), thickness=1,
                                        display_str_list=()):
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness, display_str_list)
    return np.array(image_pil)


def dataset_to_numpy_util(training_images, training_labels, training_boxes, N):
    return (training_images[:N], training_labels[:N], training_boxes[:N])


def display_digits_with_boxes(digits, predictions, labels, pred_bboxes, bboxes,
                               iou, title):
    n = min(len(digits), 10)
    indexes = np.random.choice(len(digits), size=n, replace=False)
    n_digits = digits[indexes]
    n_predictions = predictions[indexes]
    n_labels = labels[indexes]

    n_iou = []
    if len(iou) > 0:
        n_iou = iou[indexes]

    n_pred_bboxes = pred_bboxes[indexes] if len(pred_bboxes) else []
    n_bboxes = bboxes[indexes] if len(bboxes) else []

    n_digits = n_digits * 255.0
    n_digits = n_digits.reshape(n, IM_WIDTH, IM_HEIGHT)

    fig = plt.figure(figsize=(20, 4))
    plt.title(title)
    plt.yticks([])
    plt.xticks([])

    for i in range(n):
        ax = fig.add_subplot(1, 10, i + 1)
        bboxes_to_plot = []
        if len(n_pred_bboxes):
            bboxes_to_plot.append(n_pred_bboxes[i])
        if len(n_bboxes):
            bboxes_to_plot.append(n_bboxes[i])

        img_to_draw = draw_bounding_boxes_on_image_array(
            image=n_digits[i].astype("uint8"),
            boxes=np.asarray(bboxes_to_plot),
            color=["red", "green"]
        )
        plt.xticks([])
        plt.yticks([])

        ax.imshow(img_to_draw)

        if n_predictions[i] != n_labels[i]:
            ax.xaxis.label.set_color("red")

        plt.xlabel(f"True: {n_labels[i]}\nPred: {n_predictions[i]}")
        if len(n_iou):
            color = "black"
            if n_iou[i] < 0.6:
                color = "red"
            ax.xaxis.label.set_color(color)
            plt.xlabel(f"{plt.gca().get_xlabel()}\nIoU: {n_iou[i]:.2f}")


def plot_metric(history, metric_name, title):
    plt.title(title)
    plt.plot(history.history[metric_name], color="blue", label=metric_name)
    plt.plot(history.history["val_" + metric_name], color="green",
              label="val_" + metric_name)
    plt.legend()
    plt.show()


# ============================================================
# 3. LOADING AND PREPROCESSING THE DATASET
# ============================================================
def load_mnist_csv(path):
    df = pd.read_csv(path)
    labels = df["label"].values.astype("int32")
    images = df.drop(columns=["label"]).values.astype("float32")
    images = images.reshape(-1, 28, 28)
    return images, labels


def read_image_and_place_on_canvas(image, label):
    xmin = tf.random.uniform((), 0, IM_WIDTH - 28, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, IM_HEIGHT - 28, dtype=tf.int32)

    image = tf.reshape(image, (28, 28, 1))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, IM_HEIGHT, IM_WIDTH)
    image = tf.cast(image, tf.float32) / 255.0

    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)
    xmax = (xmin + 28) / IM_WIDTH
    ymax = (ymin + 28) / IM_HEIGHT
    xmin = xmin / IM_WIDTH
    ymin = ymin / IM_HEIGHT

    label = tf.one_hot(label, 10)
    box = tf.stack([ymin, xmin, ymax, xmax])
    return image, (label, box)


def build_dataset(images, labels, shuffle=False, repeat=False):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(read_image_and_place_on_canvas, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(5000, reshuffle_each_iteration=False)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def dataset_to_numpy(ds, n_batches=1):
    images_all, labels_all, boxes_all = [], [], []
    for images, (labels, boxes) in ds.take(n_batches):
        images_all.append(images.numpy())
        labels_all.append(np.argmax(labels.numpy(), axis=-1))
        boxes_all.append(boxes.numpy())
    return (np.concatenate(images_all), np.concatenate(labels_all),
            np.concatenate(boxes_all))


# ============================================================
# 4. DEFINING THE NETWORK
# ============================================================
def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, activation="relu", kernel_size=3,
                                input_shape=(IM_HEIGHT, IM_WIDTH, 1))(inputs)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, activation="relu", kernel_size=3)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(64, activation="relu", kernel_size=3)(x)
    x = tf.keras.layers.AveragePooling2D((2, 2))(x)
    return x


def dense_layers(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    return x


def classifier(inputs):
    return tf.keras.layers.Dense(10, activation="softmax", name="classification")(inputs)


def bounding_box_regression(inputs):
    return tf.keras.layers.Dense(4, name="box")(inputs)


def final_model(inputs):
    feature_cnn = feature_extractor(inputs)
    dense_output = dense_layers(feature_cnn)
    classification_output = classifier(dense_output)
    bounding_box_output = bounding_box_regression(dense_output)
    model = tf.keras.Model(inputs=inputs,
                            outputs=[classification_output, bounding_box_output])
    return model


def define_and_compile_model(inputs):
    model = final_model(inputs)
    model.compile(
        optimizer="adam",
        loss={"classification": "categorical_crossentropy", "box": "mse"},
        metrics={"classification": "accuracy", "box": "mse"}
    )
    return model


# ============================================================
# 5. INTERSECTION OVER UNION
# ============================================================
def intersection_over_union(pred_box, true_box):
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = np.split(pred_box, 4, axis=1)
    xmin_true, ymin_true, xmax_true, ymax_true = np.split(true_box, 4, axis=1)

    smoothing_factor = 1e-10

    xmin_overlap = np.maximum(xmin_pred, xmin_true)
    xmax_overlap = np.minimum(xmax_pred, xmax_true)
    ymin_overlap = np.maximum(ymin_pred, ymin_true)
    ymax_overlap = np.minimum(ymax_pred, ymax_true)

    pred_box_area = (xmax_pred - xmin_pred) * (ymax_pred - ymin_pred)
    true_box_area = (xmax_true - xmin_true) * (ymax_true - ymin_true)

    overlap_area = np.maximum(0, xmax_overlap - xmin_overlap) * \
                   np.maximum(0, ymax_overlap - ymin_overlap)
    union_area = pred_box_area + true_box_area - overlap_area

    iou = (overlap_area + smoothing_factor) / (union_area + smoothing_factor)
    return iou.flatten()


# ============================================================
# 6. MAIN EXECUTION
# ============================================================
def main():
    train_images, train_labels = load_mnist_csv(TRAIN_CSV)
    test_images, test_labels = load_mnist_csv(TEST_CSV)

    training_dataset = build_dataset(train_images, train_labels, shuffle=True, repeat=True)
    validation_dataset = build_dataset(test_images, test_labels, shuffle=False, repeat=True)

    train_digits, train_labels_np, train_boxes = dataset_to_numpy(training_dataset)
    display_digits_with_boxes(
        train_digits, train_labels_np, train_labels_np, train_boxes, train_boxes,
        np.array([]), "training digits and labels"
    )

    inputs = tf.keras.layers.Input(shape=(IM_HEIGHT, IM_WIDTH, 1))
    model = define_and_compile_model(inputs)
    model.summary()

    steps_per_epoch = len(train_images) // BATCH_SIZE
    validation_steps = len(test_images) // BATCH_SIZE

    history = model.fit(
        training_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps,
        epochs=EPOCHS
    )

    loss, classification_loss, box_loss, classification_accuracy, box_mse = \
        model.evaluate(validation_dataset, steps=1)
    print(f"\nValidation accuracy: {classification_accuracy}")

    plot_metric(history, "box_mse", "Bounding Box MSE")
    plot_metric(history, "classification_accuracy", "Classification Accuracy")

    val_digits, val_labels, val_boxes = dataset_to_numpy(validation_dataset, n_batches=1)
    predictions = model.predict(val_digits)
    predicted_labels = np.argmax(predictions[0], axis=1)
    predicted_boxes = predictions[1]

    iou = intersection_over_union(predicted_boxes, val_boxes)

    display_digits_with_boxes(
        val_digits, predicted_labels, val_labels, predicted_boxes, val_boxes,
        iou, "true and predicted values"
    )
    plt.show()


if __name__ == "__main__":
    main()
